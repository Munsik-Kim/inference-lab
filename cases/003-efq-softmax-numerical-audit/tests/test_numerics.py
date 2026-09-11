"""Independent scalar/FP64 cross-checks, plus masking and boundary tests."""
import math
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from numerics import E2M1, FOLD, PUBLIC, nearest_codes, affine_codes, dense, online


def scalar_oracle(scores, values, method, tile, block):
    """Python float64 scalar loops, explicit historical weights, NumPy PV."""
    outputs, probabilities = [], []
    for row in scores:
        maximum = -math.inf
        weights = np.zeros(len(row), dtype=np.float64)
        for begin in range(0, len(row), tile):
            end = min(begin + tile, len(row))
            newmax = max(maximum, max(row[begin:end]))
            if newmax == -math.inf:
                continue
            if maximum != -math.inf:
                weights *= math.exp(maximum - newmax)
            for b in range(begin, end, block):
                indices = [i for i in range(b, min(b + block, end)) if math.isfinite(row[i])]
                if not indices:
                    continue
                localmax = max(row[i] - newmax for i in indices)
                expn = (math.ceil((localmax - math.log(6)) / math.log(2))
                        if method == 'nearest_ceil' else
                        math.floor((localmax + math.log(2 / 9)) / math.log(2)))
                expn = max(-127, min(127, expn))
                scale = math.ldexp(1., expn)
                for i in indices:
                    x = row[i] - newmax
                    if method == 'online_fp32':
                        weight = math.exp(x)
                    elif method.startswith('nearest'):
                        exact = math.exp(x) / scale
                        code = min(range(8), key=lambda c: (abs(exact - E2M1[c]), c % 2, c))
                        weight = scale * E2M1[code]
                    else:
                        if method == 'efq_lut':
                            tau, h, cap = math.log(1 / 24), 14 / math.log(24), 15
                        else:
                            tau, h = PUBLIC[method]
                            cap = 7
                        # Written using division by a log step, independently
                        # from the tensor operand's broadcast implementation.
                        residual = x - math.log(6 * scale)
                        code = max(0, min(cap, int(math.floor((residual - tau) / (1 / h))) + 1))
                        if cap == 15:
                            code = FOLD[code]
                        weight = scale * E2M1[code]
                    weights[i] = weight
            maximum = newmax
        prob = weights / weights.sum() if weights.sum() else weights
        probabilities.append(prob)
        outputs.append(prob @ values)
    return np.asarray(outputs), np.asarray(probabilities)


class NumericalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(401)

    def test_e2m1_and_even_midpoint_ties(self):
        self.assertEqual(E2M1, (0., .5, 1., 1.5, 2., 3., 4., 6.))
        mid = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float64)
        self.assertEqual(nearest_codes(mid).tolist(), [0, 2, 2, 4, 4, 6, 6])
        self.assertEqual(nearest_codes(torch.nextafter(mid, torch.full_like(mid, -torch.inf))).tolist(), list(range(7)))
        self.assertEqual(nearest_codes(torch.nextafter(mid, torch.full_like(mid, torch.inf))).tolist(), list(range(1, 8)))
        self.assertEqual(nearest_codes(torch.tensor([-10., 0., 6., 100.])).tolist(), [0, 0, 7, 7])

    def test_affine_exact_binary_boundaries_and_clipping(self):
        z = torch.tensor([-100., -2.01, -2., -1.5, -1., -.5, 0., .5, 1., 100.], dtype=torch.float64)
        self.assertEqual(affine_codes(z, -2., 2.).tolist(), [0, 0, 1, 2, 3, 4, 5, 6, 7, 7])

    def test_public_parameters_and_lut(self):
        self.assertEqual(PUBLIC, {'efq_mmlu': (-2.9, 2.), 'efq_mean': (-3.06, 2.3), 'efq_balance': (-2.1, 2.7)})
        self.assertEqual([affine_codes(torch.tensor(0.), *p).item() for p in PUBLIC.values()], [6, 7, 6])
        self.assertEqual(FOLD, (0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7))
        tau, h = math.log(1 / 24), 14 / math.log(24)
        z = torch.tensor([tau + (j - .5) / h for j in range(16)], dtype=torch.float64)
        self.assertEqual(affine_codes(z, tau, h, True).tolist(), list(FOLD))

    def test_fp32_online_dense_fp64_reference(self):
        scores = torch.randn(2, 7, 73) * 3
        values = torch.randn(2, 73, 9)
        ref = dense(scores, values)
        result = online(scores, values, tile=19, block=8)
        exact = torch.softmax(scores.double(), -1) @ values.double()
        torch.testing.assert_close(ref['out'].double(), exact, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(result['out'], ref['out'], atol=2e-6, rtol=2e-5)

    def test_scalar_oracle_all_methods_partial_blocks(self):
        s = torch.randn(1, 5, 73) * 3
        v = torch.randn(1, 73, 11)
        s[0, 1, 16:40] = -torch.inf
        s[0, 2, 45:] = -torch.inf
        for method in ['online_fp32', 'nearest_ceil', 'nearest_efq_scale', *PUBLIC, 'efq_lut']:
            result = online(s, v, method, tile=19, block=8)
            out, prob = scalar_oracle(s[0].double().numpy(), v[0].double().numpy(), method, 19, 8)
            np.testing.assert_allclose(result['out'][0].numpy(), out, atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(result['probs'][0].numpy(), prob, atol=2e-6, rtol=2e-5)

    def test_historical_rescale_shared_operand_and_causal_mask(self):
        s = torch.zeros(1, 3, 43)
        s[..., 16:32] = 8
        s[..., 32:] = 22
        s[0, 0, 1:] = -torch.inf
        s[0, 1, 24:] = -torch.inf
        v = torch.randn(1, 43, 5)
        for method in ['online_fp32', 'nearest_ceil', 'nearest_efq_scale', *PUBLIC, 'efq_lut']:
            result = online(s, v, method, tile=16, block=8)
            torch.testing.assert_close(result['out'], result['probs'] @ v, atol=1e-6, rtol=2e-5)
            self.assertEqual(result['max_jump'][0, 2].item(), 14.)
            self.assertTrue(torch.equal(result['probs'][0, 0, 1:], torch.zeros(42)))
            ones = torch.ones_like(v)
            unity = online(s, ones, method, tile=16, block=8)
            torch.testing.assert_close(unity['out'], torch.ones_like(unity['out']), atol=1e-6, rtol=1e-6)

    def test_fully_masked_row_block_and_extreme_underflow(self):
        s = torch.full((1, 4, 67), -torch.inf)
        s[0, 1, 64:] = torch.tensor([-10000., -10001., -10002.])
        s[0, 2, :32] = -10000.
        s[0, 2, 32:64] = -20000.
        s[0, 2, 64:] = 10000.
        s[0, 3, :32] = 0.
        s[0, 3, 32:] = -10000.
        v = torch.randn(1, 67, 7)
        for method in ['online_fp32', 'nearest_ceil', 'nearest_efq_scale', *PUBLIC, 'efq_lut']:
            result = online(s, v, method, tile=32, block=16)
            self.assertFalse(result['valid'][0, 0])
            self.assertEqual(result['denom'][0, 0], 0)
            self.assertTrue((result['out'][0, 0] == 0).all())
            self.assertTrue(torch.isfinite(result['out']).all())
            self.assertTrue((result['denom'][result['valid']] > 0).all())
            if method != 'online_fp32':
                self.assertGreater(result['underflow_blocks'].item(), 0)

    def test_nonfinite_input_is_not_silently_repaired(self):
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                online(torch.tensor([[[value]]]), torch.ones(1, 1, 2))

    def test_js_analytic_cases_and_small_perturbation(self):
        from numerics import normalized_js
        p=torch.tensor([[1.,0.],[.5,.5],[.2,.8]],dtype=torch.float64)
        q=torch.tensor([[0.,1.],[.5,.5],[.2+1e-8,.8-1e-8]],dtype=torch.float64)
        result=normalized_js(p,q).tolist()
        self.assertAlmostEqual(result[0],math.log(2),places=14)
        self.assertEqual(result[1],0.)
        self.assertGreater(result[2],0.)
        # Independent longdouble direct KL oracle for a non-near-equal case.
        a=np.array([.12,.23,.65],dtype=np.longdouble)
        b=np.array([.51,.31,.18],dtype=np.longdouble)
        m=(a+b)/2
        expected=float((np.sum(a*np.log(a/m))+np.sum(b*np.log(b/m)))/2)
        got=normalized_js(torch.tensor(a.astype(float))[None],torch.tensor(b.astype(float))[None]).item()
        self.assertAlmostEqual(got,expected,places=14)

    def test_near_zero_output_is_identifiable(self):
        from numerics import metrics
        s = torch.zeros(1, 1, 2)
        v = torch.tensor([[[1., 0.], [-1., 0.]]])
        ref = dense(s, v)
        result = online(s, v, 'efq_mean')
        measured = metrics(s, v, ref, result)
        self.assertTrue(measured['near_zero_reference'].item())
        self.assertTrue(torch.isnan(measured['relative_error']).item())
        self.assertEqual(measured['abs_error'].item(), 0.)

    def test_cuda_matches_scalar_cpu_including_scale_underflow(self):
        if not torch.cuda.is_available():
            self.skipTest('GPU cross-check needs CUDA; required before freeze.')
        s = torch.randn(1, 4, 67) * 2
        s[..., 32:64] = -10000
        s[0, 0, 1:] = -torch.inf
        s[0, 1, 64:] = 10000
        v = torch.randn(1, 67, 9)
        for method in ['online_fp32', 'nearest_ceil', 'nearest_efq_scale', *PUBLIC, 'efq_lut']:
            gpu = online(s.cuda(), v.cuda(), method, tile=32, block=16)
            expected, _ = scalar_oracle(s[0].double().numpy(), v[0].double().numpy(), method, 32, 16)
            self.assertTrue(torch.isfinite(gpu['out']).all())
            np.testing.assert_allclose(gpu['out'][0].cpu().numpy(), expected, atol=3e-6, rtol=3e-5)

    def test_real_fixture_fp64_when_available(self):
        path = Path(__file__).parent / 'fixtures/qwen_dev_small.npz'
        if not path.exists():
            self.skipTest('Real fixture not collected yet; required before freeze.')
        f = np.load(path)
        q, k, v = (f[name].astype(np.float64) for name in ['q', 'k', 'v'])
        scores = q @ k.T * float(f['scaling'])
        scores[np.arange(k.shape[0])[None, :] > f['positions'][:, None]] = -np.inf
        ref = np.exp(scores - scores.max(-1, keepdims=True))
        ref /= ref.sum(-1, keepdims=True)
        result = online(torch.tensor(scores[None], dtype=torch.float32), torch.tensor(v[None], dtype=torch.float32))
        np.testing.assert_allclose(result['out'][0].numpy(), ref @ v, atol=2e-6, rtol=3e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
