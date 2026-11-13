# GPU runbook: one fine-tuning run on RunPod

Budget: ≤ 3 h of pod time. A40 48 GB ≈ $0.59/h (Secure Cloud, read 2026-11-13). **The pod bills from
creation until it is terminated (deleted), whether training or idle.**

Fill these in when the pod exists:
```bash
POD_IP=…  POD_PORT=…   # from the pod's "SSH over exposed TCP" line (root@IP -p PORT)
KEY=~/.ssh/runpod_ed25519
SSH="ssh -i $KEY -p $POD_PORT root@$POD_IP"
RS="rsync -az --info=progress2 -e 'ssh -i $KEY -p $POD_PORT'"
```

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
eval $RS -R mv train gpu requirements-train.txt data/dataset data/bank root@$POD_IP:/workspace/myvoice/
```

## 3. Set up (≈ 5–10 min: Python env, upstream code, 2.8 GB of weights from Hugging Face)
```bash
$SSH 'bash /workspace/myvoice/gpu/setup_pod.sh'
```
Ends with the GPU name and `cuda OK`.

## 4. Baseline: render the held-out verses with the untouched model (step 0)
```bash
$SSH 'bash /workspace/myvoice/gpu/sample.sh /workspace/vagdhenu/models/voice_armA_ema_2026-06-11.pt 0'
```

## 5. Train (3000 updates)
```bash
$SSH 'bash /workspace/myvoice/gpu/train.sh'
$SSH -t 'tmux attach -t train'        # watch the progress bar; Ctrl-b then d to detach
```
Closing the laptop is fine: tmux keeps training on the pod.

## 6. Watch the loss curve
TensorBoard: `https://<POD_ID>-6006.proxy.runpod.net`. Expect noisy loss that drifts down and then
flattens. Flat does **not** mean done or good; listening decides.

## 7. Listen as checkpoints arrive (every 500 updates)
```bash
eval $RS root@$POD_IP:/workspace/samples/ data/gpu_samples/
open data/gpu_samples
```
`0/` = before training, `500/`, `1000/`, … = after. The question is the same each time: does it
sound more like you, still clearly chanting the right words? Overfitting sounds like mumbling, or
fragments of training verses bleeding in.

## 8. Bring the chosen checkpoint home (1.3 GB, EMA weights only)
```bash
STEP=…   # the one that sounded best
mkdir -p models/run1 && eval $RS root@$POD_IP:/workspace/ckpt/ema_$STEP.pt models/run1/
```

## 9. Terminate the pod
In RunPod (or ask Claude): **Terminate**, not just Stop. A stopped pod still bills for its disk.
Check the pod list is empty.

## Log
| Date | GPU | Pod created → terminated | Updates | Cost | Notes |
|---|---|---|---|---|---|
| | | | | | |
