"""CLI entry point for online fine-tuning on top of a pre-trained checkpoint.

Example
-------
::

    python scripts/run_finetune.py \
        train.pretrained_path=artifacts/<run-id>/ckpt_best.pth
"""

from __future__ import annotations

from msg_embedding.training.finetune import main

if __name__ == "__main__":
    main()
