"""Failure-contract regressions with fixed synthetic numerical fixtures only."""
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source.online_v2 import (ACTIVE_FIRST_WRITE, HEADER_BYTES, NativeFloatCodec, OnlineAdapter,
                             OnlineState, PackedCodec, TerminalCode)
from codec.online import OnlineAdapter as V1Adapter


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def headers(adapter, state, **changes):
    result = state.payload.copy()
    info = adapter.terminal_info(state)
    for key, value in changes.items():
        info[key][:] = value
    adapter._headers(result, info["cursor"], info["terminal_code"], info["first_terminal_write"])
    return OnlineState(result)


class OnlineV2Tests(unittest.TestCase):
    def assert_code(self, adapter, state, expected):
        np.testing.assert_array_equal(adapter.terminal_info(state)["terminal_code"], expected)

    def test_healthy_body_matches_unchanged_v1_all_codec_modes(self):
        shape = (2,3,2)
        basis = np.broadcast_to(np.eye(3,dtype=np.float32)[:,:1],(2,3,1)).copy()
        options = [(NativeFloatCodec(shape,dtype),None,None,"transported") for dtype in ("float32","float64")]
        options += [(PackedCodec(shape,b),None,None,"transported") for b in range(2,17)]
        options += [(PackedCodec(shape,4,mixed_bits=[4,8,4]),None,None,"transported"),
                    (PackedCodec(shape,4,stochastic=True),None,None,"transported")]
        for residual in (PackedCodec(shape,4),PackedCodec(shape,8),NativeFloatCodec(shape,"float32"),NativeFloatCodec(shape,"float64")):
            options.append((PackedCodec(shape,4),residual,None,"transported"))
        options += [(PackedCodec(shape,4),PackedCodec((2,1,2),8),basis,"transported"),
                    (PackedCodec(shape,4),PackedCodec(shape,4),None,"untransported")]
        rng = np.random.default_rng(2003)
        values = rng.normal(0,.2,(4,)+shape).astype(np.float32)
        increments = rng.normal(0,.1,(5,4)+shape).astype(np.float32)
        for state_codec,residual,basis,mode in options:
            with self.subTest(bits=getattr(state_codec,"bits",None),residual=residual,mode=mode):
                old = V1Adapter(state_codec,residual,basis,mode)
                new = OnlineAdapter(state_codec,residual,basis,mode)
                a,b = old.initial(values),new.initial(values)
                np.testing.assert_array_equal(a.payload[:,8:],b.payload[:,17:])
                for increment in increments:
                    transition = lambda x: x*np.float32(.875)+increment
                    a,read_a,_ = old.step(a,transition,diagnostics=False)
                    b,read_b,_ = new.step(b,transition,diagnostics=False)
                    np.testing.assert_array_equal(a.payload[:,8:],b.payload[:,17:])
                    np.testing.assert_array_equal(read_a,read_b)
                self.assertTrue(new.terminal_info(b)["active"].all())

    def test_row_atomic_transition_failure_full_batch_and_rng_rollback(self):
        codec = PackedCodec((1,2,2),4,stochastic=True)
        adapter = OnlineAdapter(codec,PackedCodec((1,2,2),8,stochastic=True))
        state = adapter.initial(np.zeros((3,1,2,2),np.float32),seeds=np.array([11,98,9001],np.uint64))
        original = state.payload.copy()
        healthy,_,_ = adapter.step(state,lambda x:x+np.float32(.31),diagnostics=False)
        shapes = []
        def transition(x):
            shapes.append(x.shape)
            result = x+np.float32(.31);result[1]=np.inf;return result
        failed,represented,diagnostic = adapter.step(state,transition)
        self.assertEqual(shapes,[(3,1,2,2)])
        self.assert_code(adapter,failed,[0,2,0])
        np.testing.assert_array_equal(state.payload,original)
        np.testing.assert_array_equal(failed.payload[[0,2]],healthy.payload[[0,2]])
        np.testing.assert_array_equal(failed.payload[1,17:],original[1,17:])
        self.assertEqual(int(adapter.terminal_info(failed)["first_terminal_write"][1]),0)
        self.assertFalse(represented[1].any())
        self.assertEqual((diagnostic["active_at_start"],diagnostic["committed_rows"]),(3,2))
        calls = []
        original_encode = adapter._encode_rows
        def encoded(codec,values,old,rows):
            calls.append(rows.tolist());return original_encode(codec,values,old,rows)
        with patch.object(adapter,"_encode_rows",side_effect=encoded):
            again,_,_ = adapter.step(failed,lambda x:x+.2)
        self.assertEqual(calls,[[0,2],[0,2]])
        np.testing.assert_array_equal(again.payload[1,8:],failed.payload[1,8:])
        self.assertEqual(int(adapter.terminal_info(again)["cursor"][1]),2)

    def test_all_terminal_cursor_only_no_callback_decode_encode_or_revival(self):
        adapter = OnlineAdapter(PackedCodec((1,1,1),stochastic=True))
        state = adapter.initial(np.zeros((2,1,1,1),np.float32))
        state,_,_ = adapter.step(state,lambda x:np.full_like(x,np.nan))
        body = state.payload[:,8:].copy()
        def forbidden(*args,**kwargs): raise AssertionError("terminal no-op invoked work")
        with patch.object(adapter.state_codec,"decode",side_effect=forbidden), patch.object(adapter.state_codec,"encode",side_effect=forbidden):
            for _ in range(4):
                state,read,diag = adapter.step(state,forbidden)
                np.testing.assert_array_equal(state.payload[:,8:],body)
                self.assertFalse(read.any());self.assertEqual(diag["active_at_start"],0)
        np.testing.assert_array_equal(adapter.terminal_info(state)["cursor"],[5,5])
        restored = adapter.from_bytes(state.payload.tobytes(),batch_size=2)
        self.assert_code(adapter,restored,[2,2])

    def test_programming_errors_and_invalid_shapes_never_terminalize(self):
        adapter = OnlineAdapter(PackedCodec((1,2,1),stochastic=True))
        state = adapter.initial(np.zeros((2,1,2,1),np.float32))
        before = state.payload.tobytes()
        for exception in (RuntimeError("synthetic program failure"),MemoryError("synthetic OOM"),ValueError("callback bug")):
            def callback(x,exception=exception):raise exception
            with self.assertRaises(type(exception)):adapter.step(state,callback)
            self.assertEqual(state.payload.tobytes(),before)
        for value in (np.zeros((2,1,1,1)),np.zeros((2,1,2,1),complex),np.full((2,1,2,1),"bad")):
            with self.assertRaises(ValueError):adapter.step(state,lambda x,value=value:value)
        with self.assertRaises(ValueError):adapter.step(state,lambda x:x,diagnostics="yes")
        self.assertEqual(state.payload.tobytes(),before)
        self.assertTrue(adapter.terminal_info(state)["active"].all())

    def test_finite_state_range_overflow_underflow_and_explicit_clipping(self):
        values = np.array([1.,1e40,1e-300],np.float64).reshape(3,1,1,1)
        for codec in (PackedCodec((1,1,1)),NativeFloatCodec((1,1,1),"float32")):
            adapter = OnlineAdapter(codec);state = adapter.initial(np.zeros_like(values))
            after,_,_ = adapter.step(state,lambda x:values)
            self.assert_code(adapter,after,[0,3,3] if isinstance(codec,PackedCodec) else [0,3,0])
            np.testing.assert_array_equal(after.payload[1,17:],state.payload[1,17:])
        adapter = OnlineAdapter(PackedCodec((1,1,1),max_abs=1.0))
        after,represented,_ = adapter.step(adapter.initial(np.zeros_like(values)),lambda x:values)
        self.assertTrue(adapter.terminal_info(after)["active"].all())
        self.assertEqual(float(represented[1,0,0,0]),1.)

    def test_residual_encode_failure_rolls_back_tentative_state_and_rng(self):
        adapter = OnlineAdapter(PackedCodec((1,1,1),max_abs=1.,stochastic=True),PackedCodec((1,1,1),8,stochastic=True))
        state = adapter.initial(np.zeros((2,1,1,1),np.float64),seeds=np.array([19,31],np.uint64))
        values = np.array([.25,1e40],np.float64).reshape(2,1,1,1)
        after,_,_ = adapter.step(state,lambda x:values)
        self.assert_code(adapter,after,[0,5])
        np.testing.assert_array_equal(after.payload[1,17:],state.payload[1,17:])
        self.assertFalse(np.array_equal(after.payload[0,17:],state.payload[0,17:]))

    def test_adjacent_fp64_value_above_fp32_max_is_terminal_even_if_scale_rounds_finite(self):
        limit=float(np.finfo(np.float32).max)
        outside=np.nextafter(limit,np.inf)
        self.assertTrue(np.isfinite(np.float32(outside)))
        adapter=OnlineAdapter(PackedCodec((1,1,1),stochastic=True))
        state=adapter.initial(np.zeros((2,1,1,1),np.float64))
        values=np.array([limit,outside]).reshape(2,1,1,1)
        after,_,_=adapter.step(state,lambda x:values)
        self.assert_code(adapter,after,[0,3])
        np.testing.assert_array_equal(after.payload[1,17:],state.payload[1,17:])

    def test_genuine_projection_overflow_is_residual_numeric_failure(self):
        basis = np.full((1,2,1),1/np.sqrt(2),np.float32)
        adapter = OnlineAdapter(PackedCodec((1,2,1),max_abs=1.,stochastic=True),NativeFloatCodec((1,1,1),"float64"),basis)
        state = adapter.initial(np.zeros((2,1,2,1),np.float64))
        values = np.full((2,1,2,1),.5,np.float64);values[1]=1.7e308
        after,_,_ = adapter.step(state,lambda x:values)
        self.assert_code(adapter,after,[0,4])
        np.testing.assert_array_equal(after.payload[1,17:],state.payload[1,17:])

    def test_finite_component_reconstruction_overflow_terminalizes_on_step(self):
        adapter = OnlineAdapter(NativeFloatCodec((1,1,1),"float64"),NativeFloatCodec((1,1,1),"float64"))
        state = adapter.initial(np.zeros((2,1,1,1),np.float64))
        payload = state.payload.copy()
        value = np.array([1e308],dtype='<f8').view(np.uint8)
        payload[1,17:25] = value;payload[1,25:33] = value
        state = adapter.from_bytes(payload.tobytes(),batch_size=2)
        with self.assertRaisesRegex(ValueError,"ACTIVE reconstruction"):adapter.represented(state)
        seen = []
        def transition(x):seen.append(x.copy());return x+.25
        after,read,_ = adapter.step(state,transition)
        self.assert_code(adapter,after,[0,1])
        self.assertEqual(seen[0].shape,(2,1,1,1));self.assertFalse(seen[0][1].any())
        self.assertFalse(read[1].any())
        np.testing.assert_array_equal(after.payload[1,17:],payload[1,17:])

    def test_synthetic_after_write_fault_rolls_back_both_rng_counters(self):
        basis = np.array([[[1.],[0.]]],np.float32)
        adapter = OnlineAdapter(PackedCodec((1,2,1),stochastic=True),PackedCodec((1,1,1),8,stochastic=True),basis)
        state = adapter.initial(np.zeros((2,1,2,1),np.float32))
        original_expand = adapter.expand;calls = []
        def synthetic_expand(x):
            value = original_expand(x);calls.append(len(x))
            if len(calls)==2:value[1]=np.inf  # Explicit synthetic post-write fault.
            return value
        with patch.object(adapter,"expand",side_effect=synthetic_expand):
            after,read,_ = adapter.step(state,lambda x:x+.31)
        self.assert_code(adapter,after,[0,6])
        self.assertFalse(read[1].any())
        np.testing.assert_array_equal(after.payload[1,17:],state.payload[1,17:])

    def test_diagnostics_overflow_cannot_change_valid_native_or_clipped_bytes(self):
        for codec,maximum in ((NativeFloatCodec((1,2,2),"float64"),1.7e308),(PackedCodec((1,2,2),max_abs=1.),1e308)):
            adapter = OnlineAdapter(codec);state = adapter.initial(np.zeros((2,1,2,2),np.float64))
            transition = lambda x:np.full(x.shape,maximum,np.float64)
            a,ra,diagnostic = adapter.step(state,transition,diagnostics=True)
            b,rb,off = adapter.step(state,transition,diagnostics=False)
            np.testing.assert_array_equal(a.payload,b.payload);np.testing.assert_array_equal(ra,rb)
            self.assertTrue(adapter.terminal_info(a)["active"].all());self.assertEqual(off,{})
            self.assertTrue(diagnostic["diagnostic_overflow_fields"])
            json.dumps(diagnostic,allow_nan=False)

    def test_finite_wrong_label_and_nonfinite_readout_are_not_adapter_terminal_inputs(self):
        adapter = OnlineAdapter(NativeFloatCodec((1,1,1)))
        state = adapter.initial(np.zeros((2,1,1,1),np.float32))
        state,_,_ = adapter.step(state,lambda x:x+1)
        synthetic_wrong_labels = np.array([1,1])  # Gold is zero: evaluator category A.
        synthetic_logits = np.array([[np.nan,0],[0,np.inf]])  # Evaluator category B.
        self.assertTrue((synthetic_wrong_labels!=0).all());self.assertFalse(np.isfinite(synthetic_logits).all())
        state,read,_ = adapter.step(state,lambda x:x+1)
        self.assertTrue(adapter.terminal_info(state)["active"].all())
        self.assertTrue((read==2).all())

    def test_header_schema_corruption_lengths_overflow_and_legacy_rejection(self):
        adapter = OnlineAdapter(PackedCodec((1,1,1)))
        state = adapter.initial(np.zeros((1,1,1,1),np.float32))
        self.assertEqual(state.payload[0,:8].tolist(),[0]*8)
        self.assertEqual(state.payload[0,9:17].tolist(),[255]*8)
        for raw in (state.to_bytes()[:-1],state.to_bytes()+b'\0'):
            with self.assertRaises(ValueError):adapter.from_bytes(raw)
        for changed in (headers(adapter,state,terminal_code=99),headers(adapter,state,first_terminal_write=0),
                        headers(adapter,state,terminal_code=1,first_terminal_write=0),
                        headers(adapter,state,cursor=3,terminal_code=1,first_terminal_write=3)):
            with self.assertRaises(ValueError):adapter.from_bytes(changed.to_bytes())
        exhausted = headers(adapter,state,cursor=ACTIVE_FIRST_WRITE)
        with self.assertRaisesRegex(ValueError,"cursor overflow"):adapter.step(exhausted,lambda x:x)
        malformed = state.payload.copy();malformed[0,-1]=0
        with self.assertRaises(ValueError):adapter.from_payload(malformed)
        for key,value in (("format","ckda-online-v1"),("bos_policy","resume"),("header_bytes",8),
                          ("active_first_terminal_write",0),("extra_field",True)):
            cfg=json.loads(adapter.config_bytes);cfg[key]=value
            with self.assertRaises(ValueError):OnlineAdapter.from_shared(canonical(cfg))
        with self.assertRaises(ValueError):OnlineAdapter.from_shared(adapter.config_bytes+b' ')
        old = V1Adapter(PackedCodec((1,1,1)))
        with self.assertRaises(ValueError):OnlineAdapter.from_shared(old.config_bytes)
        with self.assertRaises(ValueError):adapter.step(old.initial(np.zeros((1,1,1,1),np.float32)),lambda x:x)

    def test_basis_actual_hash_bytes_shape_and_bos_identity(self):
        basis = np.array([[[1.],[0.]]],np.float32)
        adapter = OnlineAdapter(PackedCodec((1,2,1)),PackedCodec((1,1,1)),basis,bos_policy="absent")
        restored = OnlineAdapter.from_shared(adapter.config_bytes,adapter.basis_bytes)
        self.assertEqual(restored.config_bytes,adapter.config_bytes)
        self.assertEqual(json.loads(adapter.config_bytes)["basis_sha256"],hashlib.sha256(adapter.basis_bytes).hexdigest())
        different = np.array([[[0.],[1.]]],np.float32).tobytes()  # Still orthonormal; identity must reject it.
        for bad in (different,adapter.basis_bytes[:-1],adapter.basis_bytes+b'\0'):
            with self.assertRaises(ValueError):OnlineAdapter.from_shared(adapter.config_bytes,bad)
        cfg=json.loads(adapter.config_bytes);cfg['basis_shape']=[1,1,2]
        with self.assertRaises(ValueError):OnlineAdapter.from_shared(canonical(cfg),adapter.basis_bytes)
        present = OnlineAdapter(PackedCodec((1,2,1)),PackedCodec((1,1,1)),basis,bos_policy="present_write0")
        self.assertNotEqual(present.config_bytes,adapter.config_bytes)
        for chosen in (adapter,present):
            state = chosen.initial(np.zeros((1,1,2,1),np.float32))
            state,_,_ = chosen.step(state,lambda x:np.full_like(x,np.inf))
            self.assertEqual(int(chosen.terminal_info(state)['first_terminal_write'][0]),0)
            state = chosen.from_bytes(state.to_bytes())
            state,_,_ = chosen.step(state,lambda x:(_ for _ in ()).throw(AssertionError('no new BOS')))
            self.assertEqual(int(chosen.terminal_info(state)['cursor'][0]),2)

    def test_byte_ledger_payload_only_metadata_and_batched_1024_encode(self):
        adapter = OnlineAdapter(PackedCodec((2,3,2),stochastic=True))
        state = adapter.initial(np.zeros((1024,2,3,2),np.float32))
        shapes=[]
        original = adapter.state_codec.encode
        with patch.object(adapter.state_codec,'encode',wraps=original) as encode:
            state,_,_ = adapter.step(state,lambda x:shapes.append(x.shape) or x+.2,diagnostics=False)
        self.assertEqual(shapes,[(1024,2,3,2)])
        self.assertEqual(encode.call_count,1);self.assertEqual(len(encode.call_args.args[0]),1024)
        ledger=adapter.ledger(state)
        self.assertEqual(ledger['per_stream_persistent_bytes'],17+8+16+6)
        self.assertEqual(state.nbytes,len(state.payload.tobytes()))
        self.assertEqual(ledger['payload_tensor_storage_bytes'],1024*47)
        self.assertEqual(ledger['shared_bytes'],len(adapter.config_bytes)+len(adapter.basis_bytes))
        self.assertEqual(list(type(state).__dataclass_fields__),['payload'])
        with self.assertRaises((AttributeError,TypeError)):state.float_state=np.zeros(1)
        info=adapter.terminal_info(state);info['terminal_code'][0]=6
        self.assertTrue(adapter.terminal_info(state)['active'].all())

    def test_row_reordering_preserves_failure_identity_and_stochastic_streams(self):
        adapter=OnlineAdapter(PackedCodec((1,3,2),stochastic=True),PackedCodec((1,3,2),8,stochastic=True))
        initial=adapter.initial(np.zeros((4,1,3,2),np.float32),seeds=np.array([3,901,100000,2**63+1],np.uint64))
        permutation=np.array([2,0,3,1]);inverse=np.argsort(permutation)
        a=initial;b=adapter.from_payload(initial.payload[permutation])
        rng=np.random.default_rng(901)
        for step in range(6):
            increment=rng.normal(0,.3,(4,1,3,2)).astype(np.float32)
            if step==2:increment[1]=np.nan
            a,ra,_=adapter.step(a,lambda x:x*.75+increment,diagnostics=False)
            b,rb,_=adapter.step(b,lambda x:x*.75+increment[permutation],diagnostics=False)
            np.testing.assert_array_equal(a.payload,b.payload[inverse]);np.testing.assert_array_equal(ra,rb[inverse])


class FreshProcessTests(unittest.TestCase):
    def test_stochastic_restart_before_after_failure_and_arbitrary_chunks(self):
        basis=np.broadcast_to(np.array([[.6],[.8],[0]],np.float32),(2,3,1)).copy()
        adapter=OnlineAdapter(PackedCodec((2,3,2),mixed_bits=[4,8,4],stochastic=True),
                              PackedCodec((2,1,2),8,stochastic=True),basis)
        initial=adapter.initial(np.zeros((3,2,3,2),np.float32),seeds=np.array([3,211,2**63+17],np.uint64))
        increments=np.random.default_rng(991).normal(0,.15,(8,3,2,3,2)).astype(np.float32)
        increments[2,1]=np.inf;increments[4,2]=np.nan;increments[6,0]=np.inf
        states=[initial];reads=[]
        for inc in increments:
            state,represented,_=adapter.step(states[-1],lambda x:x*np.float32(.75)+inc,diagnostics=False)
            states.append(state);reads.append(represented)
        with tempfile.TemporaryDirectory(prefix='ckda-v2-restart-') as directory:
            root=Path(directory)
            shared=root/'input';shared.mkdir()
            (shared/'config.json').write_bytes(adapter.config_bytes);(shared/'basis.bin').write_bytes(adapter.basis_bytes)
            np.savez(shared/'operators.npz',increments=increments)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
            def run(start,stop,label):
                (shared/'state.bin').write_bytes(states[start].payload.tobytes())
                output=root/label
                subprocess.run([sys.executable,'-B',str(ROOT/'tests/restart_online_v2_child.py'),
                    '--input',str(shared),'--output',str(output),'--stop',str(stop)],env=env,check=True)
                self.assertEqual((output/'state.bin').read_bytes(),states[stop].payload.tobytes())
                np.testing.assert_array_equal(np.load(output/'reads.npy'),np.asarray(reads[start:stop]))
                receipt=json.loads((output/'receipt.json').read_text())
                self.assertFalse(receipt['torch_loaded']);self.assertEqual(receipt['persistent_state_fields'],['payload'])
                return output
            for split in (0,2,3,5,7):run(split,8,f'suffix{split}')
            # Each new process resumes the preceding process's actual serialized output.
            start=0
            for i,stop in enumerate((1,2,3,6,7,8)):
                if i==0:(shared/'state.bin').write_bytes(initial.payload.tobytes())
                output=root/f'chunk{i}'
                subprocess.run([sys.executable,'-B',str(ROOT/'tests/restart_online_v2_child.py'),
                    '--input',str(shared),'--output',str(output),'--stop',str(stop)],env=env,check=True)
                self.assertEqual((output/'state.bin').read_bytes(),states[stop].payload.tobytes())
                np.testing.assert_array_equal(np.load(output/'reads.npy'),np.asarray(reads[start:stop]))
                shutil.copyfile(output/'state.bin',shared/'state.bin');start=stop


if __name__ == '__main__':
    unittest.main()
