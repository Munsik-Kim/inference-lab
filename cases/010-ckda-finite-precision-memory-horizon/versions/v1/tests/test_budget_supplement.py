"""Allocation and corruption tests only; no model forward or checkpoint load."""
import copy
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"tests"))
from scripts import run_budget_supplement as runner
from scripts.audit_budget_supplement import audit_supplement
from scripts.audit_records import AuditError, canonical, sha
from test_audit_records import learned_bundle_fixture


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw = value if isinstance(value,bytes) else canonical(value)
    path.write_bytes(raw)
    return sha(raw)


def fixture(root):
    """Turn the synthetic primary fixture into a synthetic four-arm supplement."""
    from codec.packed import PackedCodec
    from codec.online import OnlineAdapter
    primary = root/"primary"
    primary.mkdir()
    learned_bundle_fixture(primary)
    out = root/"supplement"
    out.mkdir()
    policy = json.loads((ROOT/"configs/budget_supplement_v1.json").read_text())
    original = json.loads((primary/"manifest.json").read_text())
    references = {}
    for name in json.loads((primary/"protocol.json").read_text())["arms"]:
        row = json.loads((primary/"TEST"/f"{name}.json").read_text())
        references[name] = {k:row[k] for k in ("ledger","config_sha256","basis_sha256")}
        references[name]["ledger"]["payload_tensor_storage_bytes"] = 128*row["ledger"]["per_stream_persistent_bytes"]
    for name in ("protocol.json","evaluation_runtime.json","training-result.json","token_coefficients.json","token_coefficients.bin"):
        shutil.copyfile(primary/name,out/name)
    shutil.copytree(primary/"frozen-source",out/"frozen-source")
    shutil.copytree(primary/"codecs",out/"primary-codecs")
    inputs = json.loads((primary/"TEST_inputs.json").read_text())
    inputs.update(group="S3",seed=3001)
    input_hash = write(out/"TEST_inputs.json",inputs)
    scores = np.zeros((12,16),dtype=np.float64)
    np.savez(out/"calibration.npz",mixed_scores=scores)
    original["calibration_sha256"] = sha((out/"calibration.npz").read_bytes())
    primary_hash = write(out/"primary-manifest.json",original)
    policy["original_protocol_sha256"] = original["protocol_sha256"]
    policy["runtime_sha256"] = original["runtime_sha256"]
    policy["test"].update(token_sha256=inputs["token_sha256"],gold_sha256=inputs["gold_sha256"])
    ranks,maps = runner.mixed_maps(scores)
    identity = policy["seed_identities"][0]
    for key in ("checkpoint_sha256","calibration_sha256","token_table_config_sha256","token_table_data_sha256","source_sha256"):
        identity[key] = original[key]
    identity.update(test_inputs_file_sha256=input_hash,rankings=ranks.tolist(),mixed_scores_le_f64_sha256=sha(scores.astype('<f8').tobytes()))
    table_size = original["shared_token_table_bytes"]
    arms = {name:OnlineAdapter(PackedCodec((12,16,16),bits=low,mixed_bits=maps[name])) for name,low,_,_,_ in runner.SPECS}
    identity["codec_sha256"] = {name:sha(arm.config_bytes) for name,arm in arms.items()}
    for row in policy["budget_rows"]:
        n = str(row["streams"])
        cap = references[row["candidate"]]["ledger"]["total_bytes"][n]
        bit,total = max((b,references[f"UNIFORM_{b}"]["ledger"]["total_bytes"][n]) for b in range(2,17)
                        if references[f"UNIFORM_{b}"]["ledger"]["total_bytes"][n] <= cap)
        arm = arms[row["arm"]]
        shared = arm.shared_bytes+table_size
        supp = shared+int(n)*arm.bytes_per_stream
        row.update(candidate_cap_bytes=cap,highest_feasible_uniform_bits=bit,uniform_total_bytes=total,original_gap_bytes=cap-total,
            supplement_per_stream_bytes=arm.bytes_per_stream,supplement_shared_bytes=shared,supplement_total_bytes=supp,remaining_gap_bytes=cap-supp)
    csv_stream = io.StringIO(newline='')
    writer = csv.DictWriter(csv_stream,fieldnames=list(policy["budget_rows"][0]))
    writer.writeheader();writer.writerows(policy["budget_rows"])
    policy["budget_file_sha256"] = write(out/policy["budget_file"],csv_stream.getvalue().encode())
    sources = {}
    for name in ("run_budget_supplement.py","benchmark_isolated.py","audit_records.py","audit_budget_supplement.py"):
        relative = f"scripts/{name}"
        sources[relative] = write(out/"supplement-source"/relative,(ROOT/relative).read_bytes())
    policy["benchmark_helper_sha256"] = sources["scripts/benchmark_isolated.py"]
    policy_hash = write(out/"budget_supplement_v1.json",policy)
    write(out/"budget_supplement_v1.sha256",(policy_hash+'\n').encode())
    manifest = dict(schema="case010-budget-supplement-result-v1",phase=runner.PHASE,status="COMPLETE",model_seed=0,
        policy_sha256=policy_hash,family_size=630,original_family_size=546,checkpoint_sha256=identity["checkpoint_sha256"],
        source_sha256=original["source_sha256"],supplement_source_sha256=sources,same_test=True,independent_confirmation=False,
        no_calibration_refit=True,original_results_modified=False,primary_index_sha256='a'*64,primary_manifest_sha256=primary_hash,
        isolated_timing_index_sha256='b'*64,calibration_cost_index_sha256='c'*64,
        primary_dev_ledgers_sha256=write(out/"primary-dev-ledgers.json",references),complete_call_timing="NOT_MEASURED")
    write(out/"manifest.json",manifest)
    template = json.loads((primary/"TEST/UNIFORM_4.json").read_text())
    upper = 1-(.05/630)**(1/512)
    template["confidence"] = dict(family_size=630,family_alpha=.05,per_comparison_alpha=.05/630)
    for row in template["horizons"]:
        row.update(failure_upper_bound=upper,survival_lower_bound=1-upper)
    index = {}
    for name,arm in arms.items():
        row = copy.deepcopy(template)
        row.update(phase=runner.PHASE,supplement_policy_sha256=policy_hash,independent_confirmation=False,
            target_budgets=[r for r in policy["budget_rows"] if r["model_seed"] == 0 and r["arm"] == name])
        path = out/f"TEST/{name}.correctness.npz"
        path.parent.mkdir(exist_ok=True)
        shutil.copyfile(primary/"TEST/UNIFORM_4.correctness.npz",path)
        row["correctness_artifact"]["file"] = path.name
        row["config_sha256"] = write(out/f"codecs/{name}.json",arm.config_bytes)
        ledger = arm.ledger()
        ledger.update(payload_tensor_storage_bytes=512*arm.bytes_per_stream,shared_codec_bytes=arm.shared_bytes,
            shared_token_coefficient_bytes=table_size,shared_bytes=arm.shared_bytes+table_size,
            total_bytes={str(n):arm.shared_bytes+table_size+n*arm.bytes_per_stream for n in (1,16,128)})
        row["ledger"] = ledger
        digest = write(out/f"TEST/{name}.json",row)
        index[name] = dict(file=f"TEST/{name}.json",sha256=digest,execution="COMPLETE",rmst=2048.,final_survival=1.)
    write(out/"timing.json",dict(status="NOT_MEASURED"))
    digest = write(out/"index.json",dict(schema=manifest["schema"],phase=runner.PHASE,status="COMPLETE",family_size=630,
        manifest=manifest,splits={"TEST":index}))
    write(out/"index.sha256",(digest+'\n').encode())
    return out


class BudgetSupplementTests(unittest.TestCase):
    def test_fixed_maps_ties_and_scores_only(self):
        scores = np.zeros((12,16),dtype=np.float64)
        ranks,maps = runner.mixed_maps(scores)
        np.testing.assert_array_equal(ranks,np.tile(np.arange(16),(12,1)))
        self.assertEqual(list(maps),runner.NAMES)
        for name,low,high,top,_ in runner.SPECS:
            self.assertTrue(np.all(maps[name][:,:top] == high))
            self.assertTrue(np.all(maps[name][:,top:] == low))
        scores[:,15] = 2
        _,changed = runner.mixed_maps(scores)
        self.assertTrue(np.all(changed['MIXED_4_8_TOP2'][:,15] == 8))
        self.assertTrue(np.all(changed['MIXED_4_8_TOP2'][:,1] == 4))
        with self.assertRaises(TypeError):
            runner.mixed_maps(scores,test_accuracy=np.ones(4))
        for bad in (np.zeros((12,15)),np.full((12,16),np.nan),np.full((12,16),np.inf)):
            with self.assertRaises(ValueError): runner.mixed_maps(bad)

    def test_frozen_policy_prevents_arm_count_cap_and_family_changes(self):
        policy = json.loads((ROOT/"configs/budget_supplement_v1.json").read_text())
        runner.validate_policy(policy)
        self.assertEqual(sha((ROOT/"configs/budget_supplement_v1.json").read_bytes()),
                         (ROOT/"configs/budget_supplement_v1.sha256").read_text().strip())
        changes = [lambda p:p['arms'].pop(),lambda p:p['arms'][0].update(top_per_head=3),
                   lambda p:p['arms'][0]['targets'][0].update(streams=1),lambda p:p['evaluation'].update(family_size=546)]
        for change in changes:
            altered = copy.deepcopy(policy);change(altered)
            with self.assertRaises(ValueError): runner.validate_policy(altered)

    def test_real_frozen_byte_targets_without_forward(self):
        from codec.packed import PackedCodec
        from codec.online import OnlineAdapter
        evaluation = types.SimpleNamespace(PackedCodec=PackedCodec,OnlineAdapter=OnlineAdapter)
        policy = json.loads((ROOT/"configs/budget_supplement_v1.json").read_text())
        for identity in policy['seed_identities']:
            ranks = np.asarray(identity['rankings'])
            scores = np.zeros((12,16),dtype=np.float64)
            np.put_along_axis(scores,ranks,np.tile(np.arange(16,0,-1),(12,1)),axis=1)
            _,maps = runner.mixed_maps(scores)
            arms = runner.build_arms(evaluation,maps,identity,policy,identity['model_seed'],29100)
            self.assertEqual([arm.bytes_per_stream for arm in arms.values()],[1784,2072,2408,2552])
        self.assertEqual([r['remaining_gap_bytes'] for r in policy['budget_rows'][:5]],[1444,6820,676,2980,676])

    def test_imports_and_allocation_audit_do_not_load_model(self):
        code = ('import sys;sys.path.insert(0,'+repr(str(ROOT))+');'
                'from scripts import run_budget_supplement,audit_budget_supplement;'
                'assert "torch" not in sys.modules;assert "codec" not in sys.modules')
        subprocess.run([sys.executable,'-B','-c',code],check=True)

    def test_incomplete_cost_gate_rejects_before_any_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = write(root/'index.json',dict(schema='case010-isolated-timing-v1',status='STARTED',serial=True,seeds={}))
            write(root/'index.sha256',(digest+'\n').encode())
            with self.assertRaisesRegex(ValueError,'complete first'):
                runner.completed_cost_gate(root,'case010-isolated-timing-v1',{'manifest':{'model_seed':0}})


class SupplementRecordTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='ckda-budget-audit-')
        self.root = fixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def mutate_summary(self,transform):
        path = self.root/'TEST/MIXED_4_8_TOP2.json'
        row = json.loads(path.read_text());transform(row)
        digest = write(path,row)
        index = json.loads((self.root/'index.json').read_text())
        index['splits']['TEST']['MIXED_4_8_TOP2']['sha256'] = digest
        index_hash = write(self.root/'index.json',index)
        write(self.root/'index.sha256',(index_hash+'\n').encode())

    def test_complete_four_arm_audit_without_checkpoint(self):
        receipt = audit_supplement(self.root)
        self.assertEqual(receipt['verified_test_arms'],4)
        self.assertFalse(receipt['checkpoint_bytes_included'])
        self.assertEqual(receipt['family_size'],630)
        self.assertTrue(all(not Path(row['file']).is_absolute() for row in receipt['artifacts']))

    def test_runner_writes_all_four_outside_frozen_case_root(self):
        """Exercise runner I/O with synthetic summaries, never a model forward."""
        from codec.packed import PackedCodec
        from codec.online import OnlineAdapter
        primary = Path(self.temporary.name)/'ready-primary'
        shutil.copytree(self.root,primary)
        manifest = json.loads((primary/'primary-manifest.json').read_text())
        manifest['frozen_source_directory'] = 'frozen-source'
        write(primary/'manifest.json',manifest)
        for name in manifest['source_sha256']:
            write(primary/'frozen-source'/name,(primary/'frozen-source'/name.replace('/','--')).read_bytes())
        refs = json.loads((primary/'primary-dev-ledgers.json').read_text())
        shutil.rmtree(primary/'codecs')
        shutil.copytree(primary/'primary-codecs',primary/'codecs')
        for name,row in refs.items(): write(primary/'DEV'/f'{name}.json',row)
        for name in ('protocol','evaluation_runtime'):
            write(primary/f'{name}.sha256',(sha((primary/f'{name}.json').read_bytes())+'\n').encode())
        policy_path = self.root/'budget_supplement_v1.json'
        inspection = dict(directory=str(primary),manifest=manifest,protocol=json.loads((primary/'protocol.json').read_text()),
            primary_index_sha256='a'*64,manifest_sha256=sha((primary/'manifest.json').read_bytes()))
        gates = []
        for schema in ('case010-isolated-timing-v1','case010-calibration-cost-plan-v1'):
            directory = Path(self.temporary.name)/schema
            entries = {}
            for seed in range(3):
                digest = write(directory/f'seed{seed}.json',dict(model_seed=seed,checkpoint_sha256=manifest['checkpoint_sha256']))
                entries[str(seed)] = dict(file=f'seed{seed}.json',sha256=digest)
            digest = write(directory/'index.json',dict(schema=schema,status='COMPLETE',serial=True,seeds=entries))
            write(directory/'index.sha256',(digest+'\n').encode());gates.append(directory)
        inputs = json.loads((primary/'TEST_inputs.json').read_text())
        table_raw = ((primary/'token_coefficients.json').read_bytes(),(primary/'token_coefficients.bin').read_bytes())
        output = Path(self.temporary.name)/'runner-output'
        seen = []
        def fake_summarize(model,table,adapter,batch,horizons,family,seed,alpha,eps,table_bytes,correctness_output):
            self.assertEqual((family,seed,alpha,eps),(630,0,.05,(.05,.01)))
            self.assertEqual(correctness_output.parent,output/'TEST')
            name = next(n for n in runner.NAMES if n+'.correctness.npz' == correctness_output.name)
            seen.append(name)
            shutil.copyfile(self.root/'TEST'/correctness_output.name,correctness_output)
            return json.loads((self.root/'TEST'/f'{name}.json').read_text())
        fake = types.SimpleNamespace(PackedCodec=PackedCodec,OnlineAdapter=OnlineAdapter,
            torch=types.SimpleNamespace(set_num_threads=lambda n:None),
            create_model=lambda *a,**kw:types.SimpleNamespace(eval=lambda:object()),load_upstream=lambda p:None,
            token_table=lambda m:object(),table_bytes=lambda t:table_raw,
            frozen_sequences=lambda *a:object(),input_identity=lambda b:inputs,summarize_arm=fake_summarize,
            CASE_ROOT=primary/'frozen-source')
        before = {p:sha(p.read_bytes()) for p in primary.rglob('*') if p.is_file()}
        arguments = ['--primary',str(primary),'--checkpoint',str(primary/'NOT_LOADED.pt'),
            '--upstream',str(primary/'NOT_LOADED_UPSTREAM'),'--isolated-timing',str(gates[0]),
            '--calibration-cost',str(gates[1]),'--config',str(policy_path),'--output',str(output)]
        with patch.object(runner.provenance,'inspect_evaluation',return_value=inspection), \
             patch.object(runner.provenance,'load_frozen_evaluator',return_value=fake), redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(arguments),0)
        self.assertEqual(seen,runner.NAMES)
        self.assertEqual(before,{p:sha(p.read_bytes()) for p in primary.rglob('*') if p.is_file()})
        self.assertEqual(audit_supplement(output)['verified_test_arms'],4)

    def test_corrupt_tau_rejected_even_after_rehash(self):
        self.mutate_summary(lambda r:r['tau'].__setitem__(0,1))
        with self.assertRaisesRegex(AuditError,'tau disagrees'): audit_supplement(self.root)

    def test_original_family_denominator_cannot_be_reused(self):
        self.mutate_summary(lambda r:r['confidence'].update(family_size=546))
        with self.assertRaisesRegex(AuditError,'family denominator'): audit_supplement(self.root)

    def test_hidden_metadata_byte_cost_rejected(self):
        self.mutate_summary(lambda r:r['ledger'].update(shared_config_bytes=0))
        with self.assertRaisesRegex(AuditError,'shared_config_bytes mismatch'): audit_supplement(self.root)

    def test_target_gap_or_arm_omission_rejected(self):
        self.mutate_summary(lambda r:r['target_budgets'][0].update(remaining_gap_bytes=0))
        with self.assertRaisesRegex(AuditError,'target budget selection'): audit_supplement(self.root)

    def test_missing_frozen_evaluator_source_rejected(self):
        (self.root/'frozen-source/codec--online.py').unlink()
        with self.assertRaisesRegex(AuditError,'frozen source missing'): audit_supplement(self.root)


if __name__ == '__main__':
    unittest.main()
