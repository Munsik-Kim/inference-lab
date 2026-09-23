"""Technical appendix orchestration; explicitly synthetic, no model inference."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from source import run_failure_boundaries as runner
from source.online_v2 import OnlineAdapter, PackedCodec

HELPER = Path(__file__).with_name('historical_synthetic_worker.py')
spec = importlib.util.spec_from_file_location('failure_boundary_synthetic_test', HELPER)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class FailureBoundaryTests(unittest.TestCase):
    def test_four_fixed_cases_and_exact_failure_cuts(self):
        self.assertEqual(runner.CASES, ((0,'UNIFORM_2',77,838),(0,'UNIFORM_3',8,1741),
                                        (1,'UNIFORM_2',31,623),(2,'UNIFORM_2',5,867)))
        self.assertEqual(runner.cuts(1741,2048),[1740,1741,1742,2048])
        for first in [0,1,2048,True,2.5]:
            with self.assertRaises(ValueError):runner.cuts(first,2048)

    def test_actual_fresh_child_immediately_after_terminal_fault(self):
        tokens=np.ones((1,64),np.int64);tokens[0,16]=5
        gold=np.zeros(tokens.shape,np.uint8)
        adapter=OnlineAdapter(PackedCodec((1,2,2),4,stochastic=True))
        seeds=np.array([234],np.uint64)
        expected,state_expected,_=helper.synthetic_sequence_call(None,{},tokens,adapter,seeds)
        identity={'scope':'SYNTHETIC_FAULT_TEST'};ids=['synthetic-row']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            a,state,pred,counts,trace=runner.run_chunks(helper.synthetic_sequence_call,None,{},tokens,gold,adapter,
                seeds,17,root/'parent',identity,ids)
            self.assertEqual([item['group_stop'] for item in trace],[16,17,18,64])
            self.assertEqual([item['cursor'] for item in trace],[17,18,19,65])
            self.assertEqual([item['first_terminal_write'] for item in trace],[None,17,17,17])
            self.assertEqual(len({item['body_sha256'] for item in trace}),1)
            np.testing.assert_array_equal(expected,pred)
            np.testing.assert_array_equal(state_expected.payload,state.payload)
            np.savez_compressed(root/'inputs.npz',tokens=tokens.astype('u1'),gold=gold)
            appendix=root/'synthetic-appendix.json';appendix.write_text('{"synthetic_test":true}')
            plan=dict(appendix=str(appendix),appendix_sha256=runner.common.file_sha(appendix),source_sha256=runner.sources(),
                input_file=str(root/'inputs.npz'),input_file_sha256=runner.common.file_sha(root/'inputs.npz'),input_shape=[1,64],
                start_checkpoint=str(root/'parent/cut-0017'),identity=identity,sample_ids=ids,first_terminal_write=17,
                checkpoint='unused-synthetic-model',upstream='unused-synthetic-upstream',seed=0,
                output=str(root/'child'),deadline=None)
            runner.common.write_json(root/'plan.json',plan)
            result=subprocess.run([sys.executable,'-B',str(HELPER),'--failure-boundary',str(root/'plan.json')],
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            fresh=json.loads((root/'child/receipt.json').read_text())
            self.assertEqual(fresh['trace'],trace[2:])
            self.assertEqual(fresh['counts'],counts)
            self.assertEqual((root/'child/final-runtime.bin').read_bytes(),state.payload.tobytes())
            np.testing.assert_array_equal(np.load(root/'child/predictions.npy',allow_pickle=False),expected)

    def test_wrong_failure_position_and_body_mutation_rejected(self):
        adapter=OnlineAdapter(PackedCodec((1,1,1),4))
        state=adapter.initial(np.zeros((1,1,1,1),np.float32))
        for _ in range(3):state,_,_=adapter.step(state,lambda x:x+1,diagnostics=False)
        with self.assertRaisesRegex(ValueError,'failure position'):
            runner.terminal_proof(adapter,state,2,2)
        state,_,_=adapter.step(state,lambda x:np.full_like(x,np.inf),diagnostics=False)
        with self.assertRaisesRegex(ValueError,'before'):
            runner.terminal_proof(adapter,state,3,4)
        with self.assertRaisesRegex(ValueError,'body/RNG'):
            runner.terminal_proof(adapter,state,3,3,b'changed body')

    def test_appendix_freeze_binds_sources_prior_receipts_and_fixed_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);historical=root/'results/historical'
            (root/'protocol_v2.json').write_text('{"synthetic_test":true}')
            for seed,arm,row,write in runner.CASES:
                cell=historical/f'seed{seed}'/f'{runner.COHORT}--{arm}';cell.mkdir(parents=True)
                np.savez_compressed(cell/'predictions.npz',v2_full=np.zeros((1,2049),np.int8))
                record=dict(identity=dict(model_seed=seed,arm=arm),N=1,T=2048,extra=dict(original_row=row,original_first_invalid_write=write),
                    first_terminal_write=[write],status='PASS',original_invalid_case_replay=dict(same_write_as_original=True),
                    predictions_artifact=dict(sha256=runner.common.file_sha(cell/'predictions.npz')))
                runner.common.write_json(cell/'receipt.json',record)
            with patch.object(runner.common,'ROOT',root),patch.object(runner.common,'verify_protocol',return_value={}),\
                 patch.object(runner,'sources',return_value={'synthetic-source':'fixed'}):
                appendix=root/'provenance/appendix.json'
                frozen=runner.freeze(appendix,historical)
                self.assertEqual(frozen['new_memory_trials'],0)
                self.assertEqual(runner.verify_appendix(appendix),frozen)
                with self.assertRaisesRegex(ValueError,'overwrite'):runner.freeze(appendix,historical)
                tampered=json.loads(appendix.read_text());tampered['cases'][0]['first_terminal_write']+=1
                runner.common.write_json(appendix,tampered)
                appendix.with_suffix('.sha256').write_text(runner.common.file_sha(appendix)+'\n')
                with self.assertRaisesRegex(ValueError,'fixed cases'):runner.verify_appendix(appendix)
                runner.common.write_json(appendix,frozen)
                appendix.with_suffix('.sha256').write_text(runner.common.file_sha(appendix)+'\n')
                (root/frozen['cases'][0]['receipt']).write_text('changed')
                with self.assertRaisesRegex(ValueError,'reference artifact'):runner.verify_appendix(appendix)


if __name__=='__main__':unittest.main()
