"""Reuse pinned A1 CPU semantics without importing any model adapter/worker."""
from pathlib import Path
import sys
from .cpu_only import install

install()
ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT.parents[1]
A1 = CASE / 'followups/order-api-regression-v1'
for folder in [CASE / 'src', A1 / 'src']:
    sys.path.insert(0, str(folder))

from order_regression.common import canonical, digest, history, parse_slots, extract, token_hash
from order_regression.fixture import Fixture
from order_regression.registry import Registry
from order_regression.scoring import assess, correct_event, expected_total
from order_regression.audit import independent_endpoint
from order_regression.runner import run_episode

