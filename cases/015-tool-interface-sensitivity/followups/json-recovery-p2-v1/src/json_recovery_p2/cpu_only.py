"""Process-local CPU import boundary, including optional grammar dependencies."""
import importlib.abc
import sys

FORBIDDEN = {'torch', 'transformers', 'tensorflow', 'cupy', 'xgrammar'}


class CPUOnlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise ImportError('CPU_SCOPE_ONLY: forbidden dependency ' + fullname)
        return None


def install():
    loaded = sorted(FORBIDDEN.intersection(sys.modules))
    if loaded:
        raise RuntimeError('start a fresh CPU process; forbidden modules already loaded: ' + ', '.join(loaded))
    if not any(isinstance(x, CPUOnlyImports) for x in sys.meta_path):
        sys.meta_path.insert(0, CPUOnlyImports())

