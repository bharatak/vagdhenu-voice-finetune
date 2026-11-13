"""Fine-tune Vāgdhenu (IndicF5 DiT) on your clips. Runs on the GPU box; --smoke runs a 10-step trial on the Mac CPU.

    python train/finetune.py --warm ../vagdhenu/models/voice_armA_ema_2026-06-11.pt \
        --data_dir data/dataset --save_dir models/run1 --vocab ../vagdhenu/models/vocab.txt

One training step: take a batch of your clips as mel pictures, mix each with random noise by a
random amount, hide a chunk of it, and ask the model — given the text and the unhidden part — which
direction leads from the noisy picture back to the clean one. Loss = how wrong that direction was.
The optimizer nudges all 337 M weights a tiny bit (learning rate 1e-5) to be less wrong next time.

Warm start: we begin from Prathosh's trained weights, so the model already chants Sanskrit; only
the voice has to move. With ~27 min of audio, too many steps would memorise the clips (overfit),
which is why the first run is capped at 3000 updates and checkpoints are saved along the way.

Same recipe as upstream training/finetune_indicf5.py, with: EMA-aware warm start, TensorBoard
logging, an update cap, and --smoke.
"""
import argparse
import math
from pathlib import Path

CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
MELKW = dict(n_fft=1024, hop_length=256, win_length=1024, n_mel_channels=100,
             target_sample_rate=24000, mel_spec_type="vocos")
MAX_SAMPLES = 64


def warm_state(ck: dict) -> dict:
    """Checkpoint dict -> plain model weights (released files hold only EMA weights, 'ema_model.' prefixed)."""
    sd = ck.get("ema_model_state_dict") or ck.get("model_state_dict") or ck
    return {k.removeprefix("ema_model."): v for k, v in sd.items() if k not in ("initted", "step")}


def batches_per_epoch(dataset, bs: int) -> int:
    from torch.utils.data import SequentialSampler
    from f5_tts.model.dataset import DynamicBatchSampler
    return len(DynamicBatchSampler(SequentialSampler(dataset), bs, max_samples=MAX_SAMPLES))


def main(a):
    import torch
    from f5_tts.infer.utils_infer import load_model
    from f5_tts.model import DiT, Trainer
    from f5_tts.model.dataset import load_dataset

    cfm = load_model(DiT, CFG, mel_spec_type="vocos", vocab_file=a.vocab, device="cpu")
    sd = warm_state(torch.load(a.warm, map_location="cpu", weights_only=True))
    miss, unexp = cfm.load_state_dict(sd, strict=False)
    miss = [m for m in miss if "mel_spec" not in m]
    print(f"[warm-start] {len(sd)} tensors | missing (non-mel_spec): {len(miss)} | unexpected: {len(unexp)}", flush=True)
    assert not miss and not unexp, (miss[:5], unexp[:5])

    ds = load_dataset("indicf5", "custom", dataset_type="CustomDatasetPath", mel_spec_kwargs=MELKW, data_dir=a.data_dir)
    if a.smoke:
        ds.data, ds.durations = ds.data.select(range(8)), ds.durations[:8]
    per_epoch = batches_per_epoch(ds, a.bs)
    epochs = math.ceil(a.max_updates / per_epoch)
    print(f"[plan] {len(ds)} clips -> {per_epoch} updates/epoch -> {epochs} epochs ≈ {epochs * per_epoch} updates", flush=True)

    trainer = Trainer(
        cfm, epochs=epochs, learning_rate=a.lr,
        num_warmup_updates=a.warmup, save_per_updates=a.save_per, last_per_steps=a.save_per,
        checkpoint_path=a.save_dir, batch_size=a.bs, batch_size_type="frame", max_samples=MAX_SAMPLES,
        grad_accumulation_steps=1, max_grad_norm=1.0,
        logger="tensorboard", wandb_run_name=Path(a.save_dir).name,
        mel_spec_type="vocos", log_samples=False,
        accelerate_kwargs={"cpu": True} if a.smoke else {},
    )
    trainer.train(ds, num_workers=a.workers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warm", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=19200, help="audio frames per batch (1 s ≈ 94 frames)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--save_per", type=int, default=500)
    ap.add_argument("--max_updates", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="Mac CPU trial: 8 clips, 10 updates")
    a = ap.parse_args()
    if a.smoke:
        a.bs, a.save_per, a.max_updates, a.warmup, a.workers = 2000, 5, 10, 2, 2
    main(a)
