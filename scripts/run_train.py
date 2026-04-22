"""CLI entry point for MAE pre-training.

Examples
--------
Single-GPU / CPU smoke run::

    python scripts/run_train.py train.epochs=2

Multi-GPU via torchrun::

    torchrun --standalone --nproc_per_node=2 scripts/run_train.py

Resume::

    torchrun ... scripts/run_train.py train.resume=true
"""

from __future__ import annotations

from msg_embedding.training.pretrain import main

if __name__ == "__main__":
    main()
