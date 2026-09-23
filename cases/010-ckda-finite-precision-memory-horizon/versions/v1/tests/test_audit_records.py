"""Corrupt retained records/buffers to verify independent audit rejects them."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_records import (Artifacts, AuditError, HORIZONS, LEARNED_ARMS, audit_learned, canonical, check_cp_endpoint,
                                   check_online_ledger, check_summary, check_token_table,
                                   correctness, main, sha)


def summary_fixture():
    correct = np.ones((8,4),dtype=bool)
    correct[0,1] = False  # Recovery remains a first failure.
    correct[1,3] = False  # Final-position failure is not censoring.
    correct[2,0] = False
    tau = [2,4,1,None,None,None,None,None]
    endpoints = [0.650855794412824,0.7551367836334483]
    rows = [dict(horizon=t,first_failures=k,failure_probability=k/8,survival_probability=1-k/8,
                 failure_upper_bound=upper,survival_lower_bound=1-upper)
            for t,k,upper in zip([2,4],[2,3],endpoints)]
    summary = dict(n_sequences=8,max_horizon=4,sequence_ids=[f"sequence:{i}" for i in range(8)],tau=tau,
                   observed_failures=3,right_censored_sequences=5,restricted_mean_failure_free_length=3.0,
                   confidence=dict(family_size=2,family_alpha=.05,per_comparison_alpha=.025),horizons=rows,
                   step_accuracy=dict(overall=29/32,at_horizon=[dict(horizon=2,accuracy=7/8),dict(horizon=4,accuracy=7/8)],
                                      prefix=[dict(horizon=2,accuracy=14/16),dict(horizon=4,accuracy=29/32)]),
                   step_correct_counts=[7,7,8,7],step_accuracy_by_position=[7/8,7/8,1.0,7/8],
                   token_counts_at_horizons=[dict(horizon=t,prefix_correct=int(correct[:,:t].sum()),prefix_total=8*t,
                                                  final_quarter_correct=7,final_quarter_total=8) for t in [2,4]],
                   empirical_T_epsilon=[dict(epsilon=.3,horizon=2)],
                   confidence_supported_T_epsilon_lower_bound=[dict(epsilon=.3,horizon=None)])
    return summary,correct


def ledger_fixture():
    codec = dict(format="ckda-packed-v1",shape=[1,2,3],bits=4,mixed_bits=None,stochastic=False,rng=None,
                 scale_dtype="<f4",bit_order="lsb_first")
    config = dict(format="ckda-online-v1",cursor="<u8",state=codec,residual=None,basis_shape=None,
                  basis_dtype=None,gauge_bytes_per_stream=0)
    raw = canonical(config)
    # Six 4-bit codes=3B, one scale=4B, cursor=8B => 15B, independently counted.
    ledger = dict(per_stream_persistent_bytes=15,state_code_bytes=3,residual_code_bytes=0,scale_bytes=4,rng_bytes=0,
                  cursor_bytes=8,gauge_bytes=0,padding_bytes=0,shared_bytes=len(raw),shared_config_bytes=len(raw),
                  shared_basis_bytes=0,payload_tensor_storage_bytes=8*15,
                  total_bytes={str(n):len(raw)+n*15 for n in [1,16,128]})
    return ledger,raw


def learned_bundle_fixture(root):
    """A synthetic all-correct bundle; never a model experiment or study result."""
    from codec import NativeFloatCodec,PackedCodec
    from codec.online import OnlineAdapter
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        raw=value if isinstance(value,bytes) else canonical(value)
        path.write_bytes(raw)
        return sha(raw)
    shape=(12,16,16)
    basis=np.broadcast_to(np.eye(16,dtype=np.float32)[:,:4],(12,16,4)).copy()
    adapters={"NATIVE_FP32":OnlineAdapter(NativeFloatCodec(shape))}
    adapters.update({f"UNIFORM_{b}":OnlineAdapter(PackedCodec(shape,bits=b)) for b in range(2,17)})
    adapters["STOCHASTIC_4"]=OnlineAdapter(PackedCodec(shape,stochastic=True))
    for precision in (4,8,"FP32"):
        residual=NativeFloatCodec(shape) if precision == "FP32" else PackedCodec(shape,bits=precision)
        adapters[f"FULL_RESIDUAL_4_{precision}"]=OnlineAdapter(PackedCodec(shape),residual)
    for rank in (1,2,4):
        adapters[f"LOWRANK_4_8_R{rank}"]=OnlineAdapter(PackedCodec(shape),PackedCodec((12,rank,16),bits=8),basis[:,:,:rank])
    adapters["UNTRANSPORTED_4_4"]=OnlineAdapter(PackedCodec(shape),PackedCodec(shape),mode="untransported")
    adapters["MIXED_4_8"]=OnlineAdapter(PackedCodec(shape,mixed_bits=[8]*4+[4]*12))
    adapters["MIXED_6_8"]=OnlineAdapter(PackedCodec(shape,bits=6,mixed_bits=[8]*8+[6]*8))
    protocol=dict(arms=LEARNED_ARMS,splits={"TEST":dict(seed=3001,sequences=512,length=2048)},upstream_commit="a"*40)
    table_config=canonical(dict(format="ckda-token-coefficients-v2-b1-t1",token_ids=list(range(7)),
                               entries=dict(q=dict(shape=[7,2],dtype="<f4",offset=0,bytes=56))))
    table_data=b"\x00"*56
    table_size=len(table_config)+len(table_data)
    sources={}
    for name in ["codec/learned.py","codec/online.py","codec/packed.py","codec/groups.py","codec/survival.py","scripts/evaluate_learned.py"]:
        sources[name]=write(f"frozen-source/{name.replace('/','--')}",b"# synthetic retained source\n")
    manifest=dict(schema="case010-learned-evaluation-v2-canonical-token-torch",model_seed=0,phase="FROZEN_PRIMARY",test_accessed=True,checkpoint_training_status="TRAINING_COMPLETE",
                  checkpoint_updates=20000,protocol_sha256=write("protocol.json",protocol),
                  runtime_sha256=write("evaluation_runtime.json",dict(cpu_threads=1)),source_sha256=sources,
                  checkpoint_sha256="f"*64,calibration_sha256=write("calibration.npz",b"synthetic calibration fixture"),
                  token_table_config_sha256=write("token_coefficients.json",table_config),
                  token_table_data_sha256=write("token_coefficients.bin",table_data),shared_token_table_bytes=table_size,
                  upstream_commit="a"*40,required_upstream_source_sha256={"example.py":"b"*64})
    write("manifest.json",manifest)
    write("training-result.json",dict(checkpoint_sha256="f"*64,seed=0,completed_updates=20000,status="TRAINING_COMPLETE"))
    ids=[f"S3:TEST:3001:{i:06d}" for i in range(512)]
    write("TEST_inputs.json",dict(sequences=512,group_length=2048,split="TEST",sequence_ids=ids,
                                 token_sha256="c"*64,gold_sha256="d"*64))
    upper=1-(.05/546)**(1/512)
    summary=dict(n_sequences=512,max_horizon=2048,sequence_ids=ids,model_seed=0,tau=[None]*512,
                 observed_failures=0,right_censored_sequences=512,restricted_mean_failure_free_length=2048.,
                 confidence=dict(family_size=546,family_alpha=.05,per_comparison_alpha=.05/546),
                 horizons=[dict(horizon=t,first_failures=0,failure_probability=0.,survival_probability=1.,
                                failure_upper_bound=upper,survival_lower_bound=1-upper) for t in HORIZONS],
                 step_accuracy=dict(overall=1.,at_horizon=[dict(horizon=t,accuracy=1.) for t in HORIZONS],
                                    prefix=[dict(horizon=t,accuracy=1.) for t in HORIZONS]),
                 step_correct_counts=[512]*2048,step_accuracy_by_position=[1.]*2048,
                 token_counts_at_horizons=[dict(horizon=t,prefix_correct=512*t,prefix_total=512*t,
                                                final_quarter_correct=512*(t//4),final_quarter_total=512*(t//4)) for t in HORIZONS],
                 empirical_T_epsilon=[dict(epsilon=.05,horizon=2048),dict(epsilon=.01,horizon=2048)],
                 confidence_supported_T_epsilon_lower_bound=[dict(epsilon=.05,horizon=2048),dict(epsilon=.01,horizon=None)],
                 execution=dict(status="COMPLETE"),final_payload_sha256="e"*64)
    index={}
    (root/"TEST").mkdir()
    for name,adapter in adapters.items():
        row=copy.deepcopy(summary)
        correctness_path=root/f"TEST/{name}.correctness.npz"
        np.savez_compressed(correctness_path,shape=np.array([512,2048]),packed=np.full((512,256),255,dtype=np.uint8))
        row["correctness_artifact"]=dict(file=correctness_path.name,shape=[512,2048],bitorder="little",bos_included=False,
                                         sha256=sha(correctness_path.read_bytes()))
        row["config_sha256"]=write(f"codecs/{name}.json",adapter.config_bytes)
        row["basis_sha256"]=sha(adapter.basis_bytes)
        if adapter.basis_bytes:write(f"codecs/{name}.basis.bin",adapter.basis_bytes)
        ledger=adapter.ledger()
        ledger["payload_tensor_storage_bytes"]=512*adapter.bytes_per_stream
        ledger["shared_codec_bytes"]=ledger["shared_bytes"]
        ledger["shared_token_coefficient_bytes"]=table_size
        ledger["shared_bytes"]+=table_size
        ledger["total_bytes"]={str(n):ledger["shared_bytes"]+n*adapter.bytes_per_stream for n in [1,16,128]}
        row["ledger"]=ledger
        record_sha=write(f"TEST/{name}.json",row)
        index[name]=dict(file=f"TEST/{name}.json",sha256=record_sha,rmst=2048.,final_survival=1.,execution="COMPLETE")
    digest=write("index.json",dict(family_size=546,manifest=manifest,splits={"TEST":index}))
    write("index.sha256",(digest+"\n").encode())


class IndependentRecordAuditTests(unittest.TestCase):
    def verify_summary(self,summary,correct):
        return check_summary(summary,correct,expected_n=8,expected_horizon=4,horizons=[2,4],family_size=2)

    def test_independent_tau_recovery_final_failure_and_rmst(self):
        summary,correct = summary_fixture()
        result = self.verify_summary(summary,correct)
        self.assertEqual(result["first_failures"],[2,3])
        self.assertEqual(result["restricted_mean_failure_free_length"],3)

    def test_corrupted_tau_counts_and_denominators_rejected(self):
        summary,correct = summary_fixture()
        variants = []
        for key,value in [("tau",[None]*8),("n_sequences",9),("restricted_mean_failure_free_length",3.1),
                          ("step_correct_counts",[7,7,7,7]),("right_censored_sequences",6)]:
            broken=copy.deepcopy(summary);broken[key]=value;variants.append(broken)
        broken=copy.deepcopy(summary);broken["token_counts_at_horizons"][0]["prefix_total"]=32;variants.append(broken)
        broken=copy.deepcopy(summary);broken["horizons"][0]["failure_probability"]=2/16;variants.append(broken)
        broken=copy.deepcopy(summary);broken["confidence"]["family_size"]=1;variants.append(broken)
        for broken in variants:
            with self.assertRaises(AuditError):
                self.verify_summary(broken,correct)

    def test_direct_binomial_endpoint_validation(self):
        check_cp_endpoint(1-.025**(1/8),0,8,.025,"zero events")
        check_cp_endpoint(1,8,8,.025,"all events")
        check_cp_endpoint(.650855794412824,2,8,.025,"two events")
        for bad in [.5,.7,float("nan")]:
            with self.assertRaises(AuditError):
                check_cp_endpoint(bad,2,8,.025,"corrupt endpoint")
        summary,correct=summary_fixture()
        summary["horizons"][0]["failure_upper_bound"]-=.01
        with self.assertRaises(AuditError):
            self.verify_summary(summary,correct)

    def test_actual_packed_correctness_padding_hash_and_shape(self):
        _,correct=summary_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"arm.npz"
            np.savez_compressed(path,shape=np.array([8,4]),packed=np.packbits(correct,axis=1,bitorder="little"))
            store=Artifacts(directory,"relative_dataset")
            restored=correctness(store,"arm.npz",expected_shape=(8,4),expected_sha=sha(path.read_bytes()))
            np.testing.assert_array_equal(restored,correct)
            with self.assertRaises(AuditError):
                correctness(store,"arm.npz",expected_shape=(8,4),expected_sha="0"*64)
            with self.assertRaises(AuditError):
                correctness(store,"arm.npz",expected_shape=(512,2048))
            packed=np.packbits(correct,axis=1,bitorder="little");packed[0,0]|=128
            np.savez_compressed(path,shape=np.array([8,4]),packed=packed)
            with self.assertRaisesRegex(AuditError,"padding"):
                correctness(store,"arm.npz",expected_shape=(8,4))

    def test_exact_ledger_rejects_missing_cursor_and_wrong_amortization(self):
        ledger,config=ledger_fixture()
        result=check_online_ledger(ledger,config,b"",count=8)
        self.assertEqual(result["per_stream_bytes"],15)
        for key in ["cursor_bytes","scale_bytes","per_stream_persistent_bytes","shared_config_bytes","payload_tensor_storage_bytes"]:
            broken=copy.deepcopy(ledger);broken[key]-=1
            with self.assertRaises(AuditError):
                check_online_ledger(broken,config,b"",count=8)
        broken=copy.deepcopy(ledger);broken["total_bytes"]["16"]+=15
        with self.assertRaises(AuditError):
            check_online_ledger(broken,config,b"",count=8)

    def test_basis_and_token_table_actual_bytes_checked(self):
        ledger,raw=ledger_fixture()
        config=json.loads(raw)
        config["basis_shape"]=[1,2,1];config["basis_dtype"]="<f4"
        with self.assertRaisesRegex(AuditError,"basis byte count"):
            check_online_ledger(ledger,canonical(config),b"\x00"*7,count=8)
        data=np.zeros((7,2),dtype="<f4").tobytes()
        table=canonical(dict(format="ckda-token-coefficients-v1",token_ids=list(range(7)),
                             entries=dict(q=dict(shape=[7,2],dtype="<f4",offset=0,bytes=56))))
        self.assertEqual(check_token_table(table,data),len(table)+56)
        with self.assertRaises(AuditError):
            check_token_table(table,data[:-1])
        corrupt=np.zeros((7,2),dtype="<f4");corrupt[0,0]=np.inf
        with self.assertRaises(AuditError):
            check_token_table(table,corrupt.tobytes())

    def test_missing_or_corrupted_source_rejected_relative_receipts_only(self):
        with tempfile.TemporaryDirectory() as directory:
            store=Artifacts(directory,"learned_seed0")
            with self.assertRaisesRegex(AuditError,"source missing"):
                store.source_files({"codec/online.py":"0"*64})
            path=Path(directory)/"frozen-source/codec--online.py"
            path.parent.mkdir();path.write_bytes(b"# retained implementation\n")
            expected=sha(path.read_bytes())
            store.source_files({"codec/online.py":expected})
            receipt=list(store.receipts.values())[0]
            self.assertEqual(receipt["file"],"learned_seed0/frozen-source/codec--online.py")
            self.assertNotIn(directory,json.dumps(receipt))
            path.write_bytes(b"# altered implementation\n")
            with self.assertRaisesRegex(AuditError,"SHA256 mismatch"):
                store.source_files({"codec/online.py":expected})
            with self.assertRaises(AuditError):
                store.read("../outside")

    def test_output_receipts_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"receipt.json";output.write_text("original")
            with self.assertRaisesRegex(AuditError,"never overwritten"):
                main(["--toy-original",directory,"--output",str(output)])
            self.assertEqual(output.read_text(),"original")

    def test_complete_learned_artifact_route_26_arms_then_missing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            learned_bundle_fixture(root)
            receipt=audit_learned(root)
            self.assertEqual(receipt["verified_test_arms"],26)
            self.assertEqual(receipt["sequences_per_arm"],512)
            self.assertFalse(receipt["checkpoint_file_bytes_verified"])
            self.assertNotIn(directory,json.dumps(receipt))
            (root/"frozen-source/codec--online.py").unlink()
            with self.assertRaisesRegex(AuditError,"source missing"):
                audit_learned(root)


if __name__ == "__main__":
    unittest.main()
