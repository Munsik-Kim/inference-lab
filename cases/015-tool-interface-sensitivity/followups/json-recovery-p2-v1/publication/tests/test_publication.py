"""Retention and closed-manifest boundaries, using owned scratch copies only."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

PUBLICATION = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('case015_p2_publication_verifier_tests', PUBLICATION / 'verify.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class PublicationContracts(unittest.TestCase):
    def test_retained_path_rejects_escape_and_symlink_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'target').write_text('owned synthetic test')
            (root / 'link').symlink_to(root, target_is_directory=True)
            for name in ['../outside', '/absolute', 'a\\b', 'link/target']:
                with self.assertRaises(ValueError):module.safe(root, name)

    def test_map_cannot_relocate_scientific_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / module.MAP; path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'files': {'src/interface_sensitivity/model_adapter.py': {}}}))
            with self.assertRaisesRegex(ValueError, 'explicit guide allowlist'):
                module.retained(root, 'README.md')

    def test_closed_original_views_restore_exact_older_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / 'legacy'; cpu = root / 'cpu'
            old = module.restore(module.CASE, legacy)
            self.assertEqual(old['restored_files'], 457)
            module.restore(module.CASE, cpu, include_cpu=True)
            mapping = module.read(module.P2 / 'provenance/original_case_files.json')['files']
            for name, value in mapping.items():self.assertEqual(module.identity(legacy / name), value)
            for name, value in module.read(module.P2 / 'execution/cpu_original_files.json')['files'].items():
                self.assertEqual(module.identity(cpu / module.P2_REL / name), value)

    def test_current_public_bytes_and_extra_file_detection(self):
        self.assertEqual(module.verify()['status'], 'PASS')
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp) / 'case'
            shutil.copytree(module.CASE, case, ignore=shutil.ignore_patterns('__pycache__'))
            (case / 'unexpected.rpi4').write_bytes(b'synthetic forbidden payload')
            with self.assertRaisesRegex(ValueError, 'file set differs'):
                module.verify(case)


if __name__ == '__main__':unittest.main()
