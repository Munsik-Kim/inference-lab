"""CPU documentation path-contract checks; no model or GPU reproduction."""
import ast
import shlex
import tempfile
import unittest
from pathlib import Path

CASE = Path(__file__).resolve().parents[1]


def documented_outputs(markdown):
    """Read actual eval command input globs and private output roots."""
    outputs = {}
    for line in markdown.splitlines():
        if 'scripts/measure_scores.py' not in line or '--stage eval' not in line:
            continue
        args = shlex.split(line)
        pattern = args[args.index('--inputs') + 1]
        split = Path(pattern).name.removesuffix('-*.json')
        work = args[args.index('--work-dir') + 1]
        if split in outputs or not work.startswith('$CASE006_RUN/'):
            raise ValueError('Duplicate split or unexpected private root')
        outputs[split] = work.removeprefix('$CASE006_RUN/')
    if set(outputs) != {'standard', 'boundary_pool'}:
        raise ValueError('Missing or unexpected documented eval inputs')
    return outputs


def source_contract():
    """Inspect frozen writer/reader ASTs without importing or executing them."""
    writer = ast.parse((CASE / 'scripts/measure_scores.py').read_text())
    reader = ast.parse((CASE / 'supplemental/readout-ties-v1/scripts/run_readout.py').read_text())

    def assignment(tree, name):
        values = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        if len(values) != 1:
            raise ValueError('Ambiguous source assignment: ' + name)
        return values[0]

    private = assignment(writer, 'private_dir')
    if not (isinstance(private, ast.BinOp) and isinstance(private.op, ast.Div)
            and ast.unparse(private.left) == 'a.work_dir'
            and isinstance(private.right, ast.Constant)):
        raise ValueError('Unrecognized writer path')
    stage = assignment(reader, 'stage')
    if not (isinstance(stage, ast.IfExp) and isinstance(stage.test, ast.Compare)
            and ast.unparse(stage.test.left) == "item['split']"
            and len(stage.test.ops) == 1 and isinstance(stage.test.ops[0], ast.Eq)):
        raise ValueError('Unrecognized reader split selection')
    selected = ast.literal_eval(stage.test.comparators[0])
    mapping = {split: ast.literal_eval(stage.body if split == selected else stage.orelse)
               for split in ('standard', 'boundary_pool')}
    if ast.unparse(assignment(reader, 'original')) != 'a.original_run / stage':
        raise ValueError('Reader no longer resolves stage under original-run')
    reader_children = {n.right.value for n in ast.walk(reader)
                       if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
                       and isinstance(n.left, ast.Name) and n.left.id == 'original'
                       and isinstance(n.right, ast.Constant)}
    if reader_children != {private.right.value}:
        raise ValueError('Writer/reader private-vector subdirectory differs')
    return mapping, private.right.value


def check_contract(markdown, root):
    documented = documented_outputs(markdown)
    reader, child = source_contract()
    if len(set(documented.values())) != len(documented):
        raise ValueError('Distinct datasets share an output directory')
    for split, output in documented.items():
        writer_path = root / output / child
        writer_path.mkdir(parents=True, exist_ok=True)
        fixture = writer_path / (split + '--B.npz')
        fixture.write_bytes(b'DOCUMENTATION_TEST_ONLY_NOT_A_NUMPY_TENSOR')
        reader_path = root / reader[split] / child / fixture.name
        if reader_path != fixture or not reader_path.is_file():
            raise ValueError('Documentation path-contract mismatch: ' + split)


class DocumentationPathContractTests(unittest.TestCase):
    def setUp(self):
        self.markdown = (CASE / 'REPRODUCTION.md').read_text()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_documented_paths_match_actual_reader_and_writer(self):
        check_contract(self.markdown, self.root)

    def test_historical_incorrect_paths_are_rejected(self):
        paths = documented_outputs(self.markdown)
        for split, bad in [('standard', 'standard'), ('boundary_pool', 'stress')]:
            with self.subTest(split=split):
                changed = self.markdown.replace('$CASE006_RUN/' + paths[split] + '"',
                                                '$CASE006_RUN/' + bad + '"')
                self.assertNotEqual(changed, self.markdown)
                with self.assertRaisesRegex(ValueError, 'path-contract mismatch'):
                    check_contract(changed, self.root / split)

    def test_datasets_cannot_share_output_path(self):
        paths = documented_outputs(self.markdown)
        changed = self.markdown.replace('$CASE006_RUN/' + paths['boundary_pool'] + '"',
                                        '$CASE006_RUN/' + paths['standard'] + '"')
        with self.assertRaisesRegex(ValueError, 'share an output directory'):
            check_contract(changed, self.root)


if __name__ == '__main__':
    unittest.main()
