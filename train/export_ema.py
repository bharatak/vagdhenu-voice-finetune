"""Keep only the EMA weights of a training checkpoint (~1.3 GB instead of ~5 GB with optimizer state).

    python train/export_ema.py models/run1/model_last.pt models/run1/ema.pt

EMA = exponential moving average: a slow-moving copy of the weights, updated a little after each
step. It smooths out the noise of individual updates and is what you listen to / ship.
"""
import sys
from pathlib import Path

import torch


def export_ema(src: Path, dst: Path) -> None:
    ck = torch.load(src, map_location="cpu", weights_only=True)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ema_model_state_dict": ck["ema_model_state_dict"]}, dst)


if __name__ == "__main__":
    export_ema(Path(sys.argv[1]), Path(sys.argv[2]))
    print(sys.argv[2])
