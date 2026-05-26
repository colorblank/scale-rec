"""discover-main-sort GDCN+ESMM 训练入口。"""

from __future__ import annotations

import sys

from .paths import MODEL_CONFIGS
from .train_discover import main as train_discover_main


def main() -> None:
    if "--model-config" not in sys.argv:
        sys.argv.extend(["--model-config", str(MODEL_CONFIGS["discover_gdcn_esmm"])])
    train_discover_main()


if __name__ == "__main__":
    main()
