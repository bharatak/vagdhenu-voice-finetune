"""Where things live. Importing this module puts the upstream Vāgdhenu `src/` on sys.path,
so `import prep_text`, `import chant_text`, `import vagdhenu_mlx` work without editing upstream."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent           # myvoice/
UPSTREAM = Path("/Users/bharatak/Documents/vagdhenu/vagdhenu")
UPSTREAM_SRC = UPSTREAM / "src"
MODELS_UP = UPSTREAM / "models"                          # released torch weights + vocab.txt
MLX_UP = MODELS_UP / "mlx"                               # released weights converted for MLX
DATA = ROOT / "data"
INBOX = ROOT / "inbox"

if str(UPSTREAM_SRC) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SRC))
