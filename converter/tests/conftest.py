# -*- coding: utf-8 -*-
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONVERTER = REPO_ROOT / "converter"
if str(CONVERTER) not in sys.path:
    sys.path.insert(0, str(CONVERTER))
