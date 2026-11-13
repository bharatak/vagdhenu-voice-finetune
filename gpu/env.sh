# Sourced by the other gpu/*.sh scripts on the pod. Everything lives on /workspace (the pod's disk).
export W=${W:-/workspace}
export UP=${UP:-$W/vagdhenu}                # Prathosh's repo (code + released weights)
export MV=${MV:-$W/myvoice}                 # our code + dataset + bank (rsynced from the Mac)
export CK=$W/ckpt                     # training checkpoints
export EMA=$W/ema                     # EMA exports: OUTSIDE $CK (the trainer warm-starts from the newest *.pt there)
export S=$W/samples                   # rendered sample verses, one folder per step
export VAGDHENU_UPSTREAM=$UP
export PYTHONPATH=$UP/BigVGAN:$MV
export PY=${PY:-$W/venv/bin/python}
export UP_COMMIT=8bd7733              # the upstream commit we built and tested against on the Mac
