"""Synthetic restart fixture: reads only shared bytes, cache bytes and operators."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source.online_v2 import OnlineAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop", type=int, required=True)
    args = parser.parse_args()
    adapter = OnlineAdapter.from_shared((args.input/"config.json").read_bytes(), (args.input/"basis.bin").read_bytes())
    with np.load(args.input/"operators.npz", allow_pickle=False) as data:
        increments = data["increments"]
    state = adapter.from_bytes((args.input/"state.bin").read_bytes(), batch_size=increments.shape[1])
    start = int(adapter.terminal_info(state)["cursor"][0])
    reads = []
    for position in range(start, args.stop):
        state, represented, _ = adapter.step(state, lambda x: x*np.float32(.75)+increments[position], diagnostics=False)
        reads.append(represented)
    args.output.mkdir()
    (args.output/"state.bin").write_bytes(state.payload.tobytes())
    np.save(args.output/"reads.npy", np.asarray(reads))
    (args.output/"receipt.json").write_text(json.dumps(dict(start=start,stop=args.stop,torch_loaded="torch" in sys.modules,
        active=adapter.terminal_info(state)["active"].tolist(), persistent_state_fields=list(type(state).__dataclass_fields__))))


if __name__ == "__main__":
    main()
