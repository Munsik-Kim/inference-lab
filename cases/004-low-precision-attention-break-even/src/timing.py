"""Synchronized wall block then separate complete-operator CUDA event block."""
import time


def measured_block(call,synchronize,event_factory,calls,clock=time.perf_counter):
    if calls<1:raise ValueError('calls must be positive')
    synchronize();start=clock()
    for _ in range(calls):output=call()
    synchronize();wall=(clock()-start)*1000/calls
    first,last=event_factory(),event_factory()
    first.record()
    for _ in range(calls):output=call()
    last.record();last.synchronize()
    device=first.elapsed_time(last)/calls
    return wall,device
