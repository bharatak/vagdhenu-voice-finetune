# GPU runbook: one fine-tuning run on RunPod

Budget: ≤ 3 h of pod time. A40 48 GB ≈ $0.59/h (Secure Cloud, read 2026-11-13). **The pod bills from
creation until it is terminated (deleted), whether training or idle.**

When the pod exists, put its direct-SSH address ("ssh root@IP -p PORT") in `~/.ssh/config`
(zsh does not split a `$SSH` variable into words, so a config entry is simpler):
```
Host pod
  HostName <IP>
  Port <PORT>
  User root
  IdentityFile ~/.ssh/runpod_ed25519
  IdentitiesOnly yes
  ServerAliveInterval 30
```
Then `ssh pod …` and `rsync … pod:…` just work. RunPod's /workspace is a network disk that refuses
`chown`, so rsync **up** needs `--no-owner --no-group`.

## 0. One-time: an SSH key for RunPod
```bash
ssh-keygen -t ed25519 -f ~/.ssh/runpod_ed25519 -N "" -C runpod
```
Add `~/.ssh/runpod_ed25519.pub` (the **public** half) to RunPod → Settings → SSH Public Keys.
The private half never leaves the Mac.

## 1. Create the pod
1× **A40** (fallbacks: L40S or RTX 6000 Ada, 48 GB), image `runpod/pytorch` (any CUDA 12.x tag),
**60 GB** volume at `/workspace` (each full checkpoint is 5.4 GB), ports **22/tcp** + **6006/http**.

## 2. Upload code + data (~80 MB)
```bash
cd ~/Documents/vagdhenu/myvoice
rsync -azR --no-owner --no-group --exclude __pycache__ mv train gpu requirements-train.txt data/dataset data/bank pod:/workspace/myvoice/
```

## 3. Set up (≈ 5–10 min: Python env, upstream code, 2.8 GB of weights from Hugging Face)
```bash
ssh pod 'nohup bash /workspace/myvoice/gpu/setup_pod.sh > /workspace/setup.log 2>&1 < /dev/null &'
ssh pod 'tail -f /workspace/setup.log'     # Ctrl-C to stop watching
```
Ends with the GPU name and `cuda OK`.

## 4. Baseline: render the held-out verses with the untouched model (step 0)
```bash
ssh pod 'bash /workspace/myvoice/gpu/sample.sh /workspace/vagdhenu/models/voice_armA_ema_2026-06-11.pt 0'
```

## 5. Train (3000 updates)
```bash
ssh pod 'bash /workspace/myvoice/gpu/train.sh'
ssh -t pod 'tmux attach -t train'        # watch the progress bar; Ctrl-b then d to detach
```
Closing the laptop is fine: tmux keeps training on the pod. A stopped run can't be resumed (the
pinned trainer crashes on `model_last.pt`); `train.sh` refuses to start while `/workspace/ckpt` has
checkpoints, so bring home what you need and `rm -rf /workspace/ckpt` first.

## 6. Watch the loss curve
TensorBoard: `https://<POD_ID>-6006.proxy.runpod.net`. Expect noisy loss that drifts down and then
flattens. Flat does **not** mean done or good; listening decides.

## 7. After training: render every checkpoint, then listen
Training fills the A40 (44 of 46 GB), so samples are rendered afterwards (`WATCH=1 gpu/train.sh` renders
during training on a bigger GPU). Each render exports a 1.3 GB EMA copy to `/workspace/ema/`, then the
5.4 GB checkpoint can go:
```bash
ssh pod 'source /workspace/myvoice/gpu/env.sh; rm -f $CK/model_last.pt
  for ck in $CK/model_[0-9]*.pt; do bash $MV/gpu/sample.sh $ck && rm -f $ck; done'
rsync -az pod:/workspace/samples/ data/gpu_samples/
open data/gpu_samples
```
`0/` = before training, `500/`, `1000/`, … = after. The question is the same each time: does it
sound more like you, still clearly chanting the right words? Overfitting sounds like mumbling, or
fragments of training verses bleeding in.

## 7b. Fix the vocoder 'grrr' (your voice needs its own vocoder, ≈ 40 min)
Prathosh's BigVGAN vocoder was tuned only on his voice; on yours it adds a buzzy 'grrr', even when it
is handed your REAL mel. Prove it, then fine-tune it (a GAN: NVIDIA's pretrained critics, batch 4 —
batch 8 runs out of the A40's memory):
```bash
rsync -azR --no-owner --no-group data/originals_wav pod:/workspace/myvoice/   # held-out verses, as wav
ssh pod 'source /workspace/myvoice/gpu/env.sh; cd $MV
  $PY train/copysynth.py $S/copysynth data/originals_wav/*.wav        # real mel -> vocoder: grrr = vocoder
  tmux new -d -s voc "$PY train/finetune_vocoder.py --data_dir $MV/data/dataset --save_dir $W/voc \
      --eval $MV/data/originals_wav --bs 4 --steps 2000 --save_per 250 2>&1 | tee $W/voc.log"'
```
Listen to `$W/voc/eval_<step>/` against the originals (the printed mel distance went UP from 0.173
to ~0.21 while the grrr went DOWN — trust the ear). Render with it: `src/render.py … --voc $W/voc/voc_<step>.pth`.

## 8. Bring the chosen checkpoint home (1.3 GB, EMA weights only)
```bash
STEP=…   # the one that sounded best
mkdir -p models/run1 models/voc && rsync -az --info=progress2 pod:/workspace/ema/ema_$STEP.pt models/run1/
rsync -az pod:/workspace/voc/voc_<VSTEP>.pth models/voc/
PYTHONPATH=../vagdhenu/src .venv-train/bin/python -m vagdhenu_mlx.convert --voice models/run1/ema_$STEP.pt \
  --vocoder models/voc/voc_<VSTEP>.pth --vocab ../vagdhenu/models/vocab.txt --out models/mlx-myvoice
```

## 9. Terminate the pod
In RunPod (or ask Claude): **Terminate**, not just Stop. A stopped pod still bills for its disk.
Check the pod list is empty.

## Log
| Date | GPU | Pod created → terminated | Updates | Cost | Notes |
|---|---|---|---|---|---|
| 2026-11-13 | A40 48 GB, $0.59/h (CA-MTL-1) | 02:54 → ~06:45 UTC | DiT 3000 (2.0 s/step) + vocoder 2000 (1.1 s/step) | $1.60 | Chose DiT 3000 + vocoder 1250. Loss 0.84 → 0.52. Watcher off (GPU 44/46 GB). |

## Retrain later (more recordings → run 2)
1. Drop new recordings in `inbox/` as `sarga_NN.m4a` (or `sarga_NN_v11_30.m4a` for verses 11–30).
   Don't record sarga 68 — it is the held-out test.
2. `scripts/build_all.sh` → segments + content-checks only the new files, then prints the
   `mv.review … --open` command for each. Review by ear, then run `scripts/build_all.sh` again
   (rebuilds `data/dataset/` and `data/bank/`).
3. Steps 1–3 above (pod, upload, setup), then `gpu/train.sh`. Warm-start from Prathosh's model again
   (default) — not from run 1 — so the new run sees all your data at once.
4. Step 7b with `--warm models/voc/voc_1250.pth` uploaded to the pod: start the vocoder from YOUR run-1
   vocoder instead of Prathosh's.
5. Steps 8–9: bring home, convert to `models/mlx-myvoice`, terminate. Add a row to the log.
