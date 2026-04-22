from __future__ import annotations

from msg_embedding.core.types import (
    ChannelEstMode,
    JobStatus,
    JobType,
    LinkType,
    SourceType,
)


class TestEnums:
    def test_link_type_values(self):
        assert LinkType.UL == "UL"
        assert LinkType.DL == "DL"

    def test_channel_est_mode_values(self):
        assert set(ChannelEstMode) == {
            ChannelEstMode.LS,
            ChannelEstMode.MMSE,
            ChannelEstMode.IDEAL,
        }

    def test_source_type_values(self):
        assert len(SourceType) == 5
        assert SourceType.INTERNAL_SIM == "internal_sim"
        assert SourceType.FIELD == "field"

    def test_job_status_values(self):
        assert len(JobStatus) == 5
        assert JobStatus.PENDING == "pending"
        assert JobStatus.COMPLETED == "completed"

    def test_job_type_values(self):
        assert len(JobType) == 8
        assert JobType.TRAIN == "train"
        assert JobType.BRIDGE == "bridge"

    def test_enums_are_strings(self):
        assert isinstance(LinkType.UL, str)
        assert isinstance(SourceType.FIELD, str)
        assert isinstance(JobStatus.RUNNING, str)
