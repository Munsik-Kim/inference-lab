"""Scalar sufficient statistics from actual reloaded FP32 matrices lifted to FP64."""
import numpy as np

def squared(a):return float(np.sum(a*a,dtype=np.float64))
def dot(a,b):return float(np.sum(a*b,dtype=np.float64))
def measure(w,x,z,delta,what,what0,teacher_run=None,candidate_run=None,what_run=None):
 w,x,z,delta,what,what0=(np.asarray(a,np.float64) for a in (w,x,z,delta,what,what0))
 A=w@(x-z);C=delta@z;B=(what-w-delta)@z;B0=(what0-w)@z
 R=w@x-what@z;R0=w@x-what0@z;dB=B-B0
 ss={f'{n}_norm2':squared(a) for n,a in [('A',A),('C',C),('B',B),('B0',B0),('deltaB',dB)]}
 ss.update(dot_A_B0=dot(A,B0),dot_A_C=dot(A,C),dot_B0_C=dot(B0,C),dot_R0_C=dot(R0,C),dot_R0minusC_deltaB=dot(R0-C,dB))
 ss.update(SSE_before=squared(R0),SSE_after=squared(R),reference_SSE=squared(w@x),reference_elements=int(R.size),fixedB_SSE=squared(A-C-B0),continuous_SSE=squared(A-C))
 ss['T_inherited']=ss['C_norm2']-2*ss['dot_A_C'];ss['T_cancellation']=2*ss['dot_B0_C']
 ss['T_cont']=ss['C_norm2']-2*ss['dot_R0_C'];ss['T_requant']=ss['deltaB_norm2']-2*ss['dot_R0minusC_deltaB']
 ss['Delta']=ss['SSE_after']-ss['SSE_before'];ss['decomposition_gap']=ss['Delta']-ss['T_inherited']-ss['T_cancellation']-ss['T_requant']
 ss['identity_scale']=max(1,abs(ss['SSE_after']),abs(ss['SSE_before']),abs(ss['T_cont']),abs(ss['T_requant']))
 ss['identity_relative_gap']=abs(ss['decomposition_gap'])/ss['identity_scale']
 ss['residual_identity_relative']=np.linalg.norm(R-(A-C-B))/max(1,np.linalg.norm(R))
 valid=(ss['reference_SSE']/R.size)**.5>1e-6
 ss['normalized_SSE']=ss['SSE_after']/ss['reference_SSE'] if valid else None
 ss['relative_Frobenius']=ss['normalized_SSE']**.5 if valid else None
 ss['cosine_A_B0']=dot(A,B0)/(squared(A)*squared(B0))**.5 if squared(A)*squared(B0)>0 else None
 ss['arithmetic_residual_norm2']=ss['BF16_output_residual_norm2']=ss['cast_residual_norm2']=None
 if teacher_run is not None:
  rrun=np.asarray(teacher_run,np.float64)-np.asarray(candidate_run,np.float64)
  ss['BF16_output_residual_norm2']=squared(rrun);ss['arithmetic_residual_norm2']=squared(rrun-R)
 if what_run is not None:ss['cast_residual_norm2']=squared((np.asarray(what_run,np.float64)-what)@z)
 return {k:float(v) if isinstance(v,np.floating) else v for k,v in ss.items()}
