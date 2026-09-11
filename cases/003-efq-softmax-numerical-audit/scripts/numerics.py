"""Independent FP32 simulation of paper v1, not a packed FP4 kernel."""
import math
import torch
import torch.nn.functional as F

E2M1 = (0., .5, 1., 1.5, 2., 3., 4., 6.)
FOLD = (0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7)
PUBLIC = {'efq_mmlu': (-2.90, 2.00), 'efq_mean': (-3.06, 2.30),
          'efq_balance': (-2.10, 2.70)}


def nearest_codes(x):
    """Nearest E2M1, exact midpoint ties choose the even bit-pattern index."""
    bounds = x.new_tensor((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.))
    c = torch.bucketize(x.contiguous(), bounds, right=False)
    ties = (x == bounds[c.clamp(max=6)]) & (c < 7) & (c % 2 == 1)
    return c + ties.long()


def affine_codes(z, tau, h, lut=False):
    hi = 15 if lut else 7
    c = (torch.floor((z - tau) * h) + 1).clamp(0, hi).long()
    return torch.tensor(FOLD, device=z.device)[c] if lut else c


def operand(x, method, block=32, tau=None, h=None):
    """x: [head, sampled query, tile keys]; blocks never cross rows."""
    size = x.shape[-1]
    xb = F.pad(x, (0, (-size) % block), value=-torch.inf)
    xb = xb.reshape(*x.shape[:-1], -1, block)
    live = torch.isfinite(xb)
    m = xb.amax(-1, keepdim=True)
    if method == 'nearest_ceil':
        # Independent headroom scale, not claimed to match the paper's MXFP4.
        raw_k = torch.ceil((m - math.log(6)) / math.log(2))
    else:
        raw_k = torch.floor((m + math.log(2 / 9)) / math.log(2))
    finite_block = torch.isfinite(m)
    k = torch.where(finite_block, raw_k, -127.).clamp(-127, 127)
    scale = torch.ldexp(torch.ones_like(k), k.int())
    if method.startswith('nearest'):
        # Explicit exponential followed by quantization; same scale control
        # uses exactly Eq. 8 as EFQ. No exponent-only shortcut here.
        exact = xb.exp()
        u = exact / scale
        c = nearest_codes(u)
    else:
        z = xb - k * math.log(2) - math.log(6)
        if method == 'efq_lut':
            tau, h = math.log(1 / 24), 14 / math.log(24)
        elif method in PUBLIC:
            tau, h = PUBLIC[method]
        elif method != 'efq_calibrated':
            raise ValueError(method)
        if torch.is_tensor(tau):
            tau, h = tau[..., None], h[..., None]
        c = affine_codes(z, tau, h, method == 'efq_lut')
    c = torch.where(live, c, 0)
    p = x.new_tensor(E2M1)[c] * scale
    p = torch.where(live, p, 0.)
    shape = (*x.shape[:-1], -1)
    return (p.reshape(shape)[..., :size],
            (c == 0).reshape(shape)[..., :size],
            ((raw_k < -127) & finite_block).sum(),
            ((scale == 0) & finite_block).sum())


def dense(scores, values):
    valid = torch.isfinite(scores).any(-1)
    m = scores.amax(-1, keepdim=True)
    shift = scores - torch.where(valid[..., None], m, 0.)
    weights = shift.exp()
    denom = weights.sum(-1)
    p = weights / torch.where(valid, denom, 1.)[..., None]
    return {'out': p @ values, 'probs': p, 'denom': denom, 'valid': valid,
            'code_zero': weights == 0, 'underflow_blocks': scores.new_tensor(0.),
            'zero_scales': scores.new_tensor(0.),
            'max_jump': torch.zeros_like(denom)}


def online(scores, values, method='online_fp32', tile=128, block=32,
           tau=None, h=None, probabilities=True):
    """One shared reconstructed operand updates both accumulators.

    Scores/QK and values are unchanged between methods. All-masked rows return
    zero and valid=False; valid rows with a zero denominator remain NaN.
    """
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise ValueError('NaN or positive infinity in input scores')
    valid = torch.isfinite(scores).any(-1)
    m = scores.new_full(scores.shape[:-1], -torch.inf)
    a = scores.new_zeros(*scores.shape[:-1], values.shape[-1])
    ell = torch.zeros_like(m)
    max_jump = torch.zeros_like(m)
    blocks, maxima, zeros = [], [], []
    underflows = scores.new_tensor(0.)
    zero_scales = scores.new_tensor(0.)
    for start in range(0, scores.shape[-1], tile):
        s = scores[..., start:start + tile]
        new_m = torch.maximum(m, s.amax(-1))
        finite = torch.isfinite(new_m)
        alpha = torch.where(torch.isfinite(m), (m - new_m).exp(), 0.)
        jump = torch.where(torch.isfinite(m) & finite, new_m - m, 0.)
        max_jump = torch.maximum(max_jump, jump)
        x = s - torch.where(finite, new_m, 0.)[..., None]
        if method == 'online_fp32':
            p, zero = x.exp(), x.exp() == 0
        else:
            p, zero, uf, zs = operand(x, method, block, tau, h)
            underflows += uf
            zero_scales += zs
        # There is intentionally no separately reconstructed denominator path.
        a = a * alpha[..., None] + p @ values[..., start:start + tile, :]
        ell = ell * alpha + p.sum(-1)
        if probabilities:
            blocks.append(p)
            maxima.append(new_m)
            zeros.append(zero)
        m = new_m
    out = a / torch.where(valid, ell, 1.)[..., None]
    result = {'out': out, 'denom': ell, 'valid': valid,
              'underflow_blocks': underflows, 'zero_scales': zero_scales,
              'max_jump': max_jump}
    if probabilities:
        # Each tile was formed relative to its own running maximum. Restore
        # its effective contribution at the final maximum before JS/top-k.
        weights = torch.cat([p * torch.where(torch.isfinite(mt), (mt - m).exp(), 0.)[..., None]
                             for p, mt in zip(blocks, maxima)], dim=-1)
        total = weights.sum(-1)
        result['probs'] = weights / torch.where(valid, total, 1.)[..., None]
        result['code_zero'] = torch.cat(zeros, dim=-1)
    return result


def normalized_js(p, q):
    """FP64 JS with explicit normalization and a stable nonnegative form.

    For |r|<.01 the even expansion through r**8 has remainder <1.2e-22.
    This only affects the diagnostic; the simulated attention stays FP32.
    """
    p, q = p.double(), q.double()
    p = p / torch.where(p.sum(-1, keepdim=True) > 0, p.sum(-1, keepdim=True), 1.)
    q = q / torch.where(q.sum(-1, keepdim=True) > 0, q.sum(-1, keepdim=True), 1.)
    total = p + q
    r = (p-q) / torch.where(total > 0, total, 1.)
    r2 = r.square()
    series = r2/2 + r2.square()/12 + r2.pow(3)/30 + r2.pow(4)/56
    plus, minus = 1+r, 1-r
    exact = (plus * torch.where(plus > 0, torch.log1p(r), 0.) + minus * torch.where(minus > 0, torch.log1p(-r), 0.))/2
    return (total/2 * torch.where(r.abs() < .01, series, exact)).sum(-1)


def metrics(scores, values, ref, result, k=8, norm_floor=1e-6):
    """Per-row diagnostics; unit Frobenius norms are calculated separately."""
    live = torch.isfinite(scores)
    n = live.sum(-1)
    p, q = ref['probs'], result['probs']
    o, r = result['out'], ref['out']
    rn = torch.linalg.vector_norm(r, dim=-1)
    err = torch.linalg.vector_norm(o - r, dim=-1)
    near = rn <= norm_floor * math.sqrt(r.shape[-1])
    rel = torch.where(~near, err / rn, torch.nan)
    on = torch.linalg.vector_norm(o, dim=-1)
    cosine = torch.where((rn > 0) & (on > 0), (o * r).sum(-1) / (rn * on), torch.nan)
    js = normalized_js(p, q)
    entropy = -(p * torch.where(p > 0, p.log(), 0.)).sum(-1)
    lo = torch.where(live, scores, torch.inf).amin(-1)
    spread = scores.amax(-1) - lo
    best = torch.sort(scores, descending=True, stable=True).values[..., :2]
    gap = torch.where(n > 1, best[..., 0] - best[..., 1], torch.nan)
    kk = min(k, scores.shape[-1])
    # Stable sorting resolves equal weights by ascending key index.
    pi = torch.argsort(p, descending=True, stable=True)[..., :kk]
    qi = torch.argsort(q, descending=True, stable=True)[..., :kk]
    rank_valid = torch.arange(kk, device=scores.device) < n[..., None]
    matches = (pi[..., :, None] == qi[..., None, :]) & rank_valid[..., :, None] & rank_valid[..., None, :]
    overlap = matches.any(-1).sum(-1) / n.clamp(max=kk)
    sorted_q = torch.sort(torch.where(live, q, -1.), stable=True).values
    adjacent_live = (sorted_q[..., 1:] >= 0) & (sorted_q[..., :-1] >= 0)
    ties = ((sorted_q[..., 1:] == sorted_q[..., :-1]) & adjacent_live).sum(-1)
    tie_fraction = torch.where(n > 1, ties / (n - 1), 0.)
    z = result['code_zero'] & live
    return {'abs_error': err, 'relative_error': rel, 'near_zero_reference': near,
            'reference_norm': rn, 'cosine': cosine, 'js_nats': js,
            'top8_overlap': overlap, 'effective_probability_tie_fraction': tie_fraction,
            'e2m1_zero_fraction': z.sum(-1) / n,
            'removed_reference_mass': (p * z).sum(-1),
            'score_spread': spread, 'entropy_nats': entropy, 'top_logit_gap': gap,
            'max_row_max_jump': result['max_jump'],
            'denominator': result['denom'], 'valid': result['valid']}
