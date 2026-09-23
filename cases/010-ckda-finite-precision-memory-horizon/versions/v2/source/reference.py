"""Load the single, unchanged v1 reference source copy under a distinct package root."""
from pathlib import Path
import sys
import importlib.util
ROOT=Path(__file__).resolve().parent/'v1_reference'
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from codec import packed,online,groups,survival,learned
spec=importlib.util.spec_from_file_location('case010_v1_evaluator',ROOT/'scripts/evaluate_learned.py')
v1=importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
