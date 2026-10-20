"""CPU-only Prometheus interval arithmetic, not per-request attribution."""
import re


def metric_scalar(text, name):
    values = []
    for line in text.splitlines():
        if line.startswith('#'):
            continue
        match = re.match(r'^([^\s{]+)(?:\{[^}]*\})?\s+([^\s]+)', line)
        if match and match[1] == name:
            values.append(float(match[2]))
    return sum(values) if values else None


def difference(before, after, name):
    a, b = metric_scalar(before, name), metric_scalar(after, name)
    if a is None or b is None:
        return None
    if b < a:
        raise ValueError('Counter reset inside block: ' + name)
    return b - a


def interval_mean(before, after, base):
    total, count = difference(before, after, base + '_sum'), difference(before, after, base + '_count')
    return total / count if total is not None and count else None
