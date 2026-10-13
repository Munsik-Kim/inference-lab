"""Small CPU contracts, not substitutes for the required model diagnostics."""
import json
import random
import unittest
from contracts import token_hash, tool_history, common_length, select_projection


class DiagnosticContracts(unittest.TestCase):
    def test_error_history_keeps_calls_not_explanatory_response(self):
        initial = [{'role': 'user', 'content': 'fixture'}]
        calls = [{'name': 'ApplyFixturePatch', 'arguments': {'enabled': True, 'limit': 14}}]
        errors = [{'error': 'unknown tool/incorrect argument keys'}]
        history = tool_history(initial, calls, errors)
        self.assertEqual(history[1]['content'], '')
        self.assertEqual(history[1]['tool_calls'][0]['function']['name'], 'ApplyFixturePatch')
        self.assertEqual(json.loads(history[2]['content']), errors[0])
        self.assertEqual(initial, [{'role': 'user', 'content': 'fixture'}])

    def test_common_prefix_has_identical_prior_token_bytes(self):
        correct = [151644, 2, 3, 12, 13]
        wrong = [151644, 2, 3, 99, 100, 101]
        n = common_length(correct, wrong)
        self.assertEqual(n, 3)
        self.assertEqual(token_hash(correct[:n]), token_hash(wrong[:n]))
        self.assertNotEqual(token_hash(correct), token_hash(wrong))

    def test_selection_uses_two_context_error_and_matching_cost(self):
        names = [f'model.layers.{i // 7}.mlp.' + ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'][i % 7] for i in range(252)]
        errors = {key: {n: {'baseline_energy': 2., 'difference_sse': 1.} for n in names} for key in ('AGENT-6', 'AGENT-7')}
        target = names[6]
        for key in errors:
            errors[key][target]['difference_sse'] = 10.
        errors['AGENT-7'][names[5]]['baseline_energy'] = 0
        metadata = {'files': [{'layer': n, 'shape': [8, 64], 'bytes': 400} for n in names]}
        q = select_projection(errors, metadata)
        self.assertEqual(q['target'], target)
        self.assertIn(names[5], q['zero_energy_excluded'])
        self.assertEqual(q['control'].split('.')[-1], 'down_proj')
        self.assertNotEqual(q['control'].split('.')[2], target.split('.')[2])
        self.assertEqual(q['nominal_storage_increase_bytes'], 624)
        self.assertEqual(q['control'], random.Random(610132).choice(q['matched_control_candidates']))

    def test_full_projection_restore_removes_beta_and_preserves_other_module(self):
        import torch
        from intervention import restore_projection
        class Scaled(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in_features = self.out_features = 2
                self.weight = torch.nn.Parameter(torch.eye(2))
                self.register_buffer('beta_inverse', torch.tensor([2., 3.]))
            def forward(self, x):
                return torch.nn.functional.linear(x * self.beta_inverse, self.weight)
        model = torch.nn.Module(); model.block = torch.nn.Module()
        model.block.q_proj = Scaled(); model.block.k_proj = Scaled()
        unchanged = model.block.k_proj
        original = torch.nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            original.weight.copy_(torch.eye(2)); original.bias.copy_(torch.tensor([4., 5.]))
        restore_projection(model, 'block.q_proj', original, 'cpu')
        self.assertFalse(hasattr(model.block.q_proj, 'beta_inverse'))
        self.assertTrue(torch.equal(model.block.q_proj(torch.ones(1, 2)), torch.tensor([[5., 6.]])))
        self.assertIs(model.block.k_proj, unchanged)


if __name__ == '__main__':
    unittest.main()
