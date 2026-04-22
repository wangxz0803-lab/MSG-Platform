from __future__ import annotations

import torch
import torch.nn as nn

from msg_embedding.models.ema import SimpleEMA


class TestSimpleEMA:
    def test_register_captures_params(self):
        model = nn.Linear(4, 2)
        ema = SimpleEMA(model, decay=0.99)
        assert len(ema.shadow) == 2  # weight + bias

    def test_update_changes_shadow(self):
        model = nn.Linear(4, 2)
        ema = SimpleEMA(model, decay=0.5)
        old_weight_shadow = ema.shadow["weight"].clone()

        with torch.no_grad():
            model.weight.fill_(1.0)
        ema.update(model)

        assert not torch.equal(ema.shadow["weight"], old_weight_shadow)

    def test_apply_and_restore(self):
        model = nn.Linear(4, 2)
        ema = SimpleEMA(model, decay=0.5)

        with torch.no_grad():
            model.weight.fill_(99.0)
        ema.update(model)

        ema.apply_shadow(model)
        assert not torch.equal(model.weight.data, torch.full_like(model.weight.data, 99.0))

        ema.restore(model)
        torch.testing.assert_close(model.weight.data, torch.full_like(model.weight.data, 99.0))
