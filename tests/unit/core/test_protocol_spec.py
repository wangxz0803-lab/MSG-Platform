from __future__ import annotations

from msg_embedding.core.protocol_spec import PROTOCOL_SPEC


class TestProtocolSpec:
    def test_has_expected_keys(self):
        expected = {
            "cqi",
            "srs_sinr",
            "rsrp",
            "rsrp_srs",
            "rsrp_cb",
            "cell_rsrp",
            "path_loss",
            "tx_ant_num",
            "rx_ant_num",
            "ue_tx_ant_num",
            "srs_hopping_rb",
            "pdp",
            "srs_w1",
            "srs_w2",
            "srs_w3",
            "srs_w4",
            "srs_cb_sinr",
            "ta",
        }
        assert set(PROTOCOL_SPEC.keys()) == expected

    def test_all_entries_have_min_max(self):
        for key, spec in PROTOCOL_SPEC.items():
            assert "min" in spec, f"{key} missing 'min'"
            assert "max" in spec, f"{key} missing 'max'"
            assert spec["min"] < spec["max"], f"{key}: min >= max"

    def test_cqi_discrete_range(self):
        cqi = PROTOCOL_SPEC["cqi"]
        assert cqi["min"] == 0
        assert cqi["max"] == 15
        assert cqi["type"] == "discrete"

    def test_rsrp_db_domain(self):
        for key in ("rsrp", "rsrp_srs", "rsrp_cb", "cell_rsrp"):
            spec = PROTOCOL_SPEC[key]
            assert spec["min"] == -160.0
            assert spec["max"] == -60.0
            assert spec["process"] == "db"

    def test_sinr_linear_range(self):
        for key in ("srs_sinr", "srs_cb_sinr"):
            spec = PROTOCOL_SPEC[key]
            assert spec["process"] == "linear"
            lr = spec["linear_range"]
            assert isinstance(lr, tuple)
            assert len(lr) == 2
            assert lr[0] < lr[1]

    def test_srs_weights_unit_range(self):
        for i in range(1, 5):
            spec = PROTOCOL_SPEC[f"srs_w{i}"]
            assert spec["min"] == 0.0
            assert spec["max"] == 1.0
