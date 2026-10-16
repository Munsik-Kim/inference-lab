"""Reuse pinned CPU parser, history and identity primitives, not P1 scoring."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT.parents[1]
sys.path.insert(0, str(P1 / 'src'))
from interface_sensitivity.common import atomic, read, rows, digest, canonical, sha, require, utc, token_hash, history, extract

METHODS = ('BF16', 'D16', 'RTN')
APIS = ('API_V1', 'API_V2')
SCORER = 'case015-a1-verified-workflow-v1'


def parse_slots(raw):
    # Same first-valid-object parser as P1; retain original indices and malformed
    # marker attempts rather than treating them as executed calls.
    return [{'call_index':i, 'parsed':extract(s.split('</tool_call>')[0])}
            for i,s in enumerate(raw.split('<tool_call>')[1:])]


def same(a, b):
    return canonical(a) == canonical(b)
