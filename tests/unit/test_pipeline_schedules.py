"""Tests for pipeline schedule selection."""


from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.distributed.pipeline_schedules import (
    PIPELINE_SCHEDULE_SPECS,
    PipelineConfig,
    PipelineScheduleFactory,
    PipelineScheduleSpec,
    PipelineScheduleType,
)


class TestPipelineScheduleType:
    """Tests for PipelineScheduleType enum."""

    def test_all_types(self):
        expected = {"gpipe", "interleaved_1f1b", "zero_bubble", "zbv_zero_bubble", "looped_bfs"}
        actual = {t.value for t in PipelineScheduleType}
        assert actual == expected

    def test_specs_cover_all_types(self):
        for stype in PipelineScheduleType:
            assert stype in PIPELINE_SCHEDULE_SPECS


class TestPipelineScheduleSpec:
    """Tests for PipelineScheduleSpec."""

    def test_gpipe_spec(self):
        spec = PIPELINE_SCHEDULE_SPECS[PipelineScheduleType.GPIPE]
        assert spec.display_name == "GPipe"
        assert spec.requires_async is False
        assert spec.min_stages == 2

    def test_zero_bubble_requires_async(self):
        spec = PIPELINE_SCHEDULE_SPECS[PipelineScheduleType.ZERO_BUBBLE]
        assert spec.requires_async is True

    def test_zbv_requires_4_stages(self):
        spec = PIPELINE_SCHEDULE_SPECS[PipelineScheduleType.ZBV_ZERO_BUBBLE]
        assert spec.min_stages == 4
        assert spec.requires_async is True

    def test_interleaved_1f1b(self):
        spec = PIPELINE_SCHEDULE_SPECS[PipelineScheduleType.INTERLEAVED_1F1B]
        assert spec.requires_async is False
        assert spec.min_stages == 2

    def test_looped_bfs(self):
        spec = PIPELINE_SCHEDULE_SPECS[PipelineScheduleType.LOOPED_BFS]
        assert spec.requires_async is False


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_defaults(self):
        config = PipelineConfig()
        assert config.schedule == PipelineScheduleType.INTERLEAVED_1F1B
        assert config.num_stages == 2
        assert config.num_microbatches == 4

    def test_to_dict(self):
        config = PipelineConfig(
            schedule=PipelineScheduleType.ZERO_BUBBLE,
            num_stages=4,
            num_microbatches=8,
        )
        d = config.to_dict()
        assert d["schedule"] == "zero_bubble"
        assert d["num_stages"] == 4
        assert d["num_microbatches"] == 8


class TestPipelineScheduleFactory:
    """Tests for PipelineScheduleFactory."""

    def test_cuda_hopper_supports_zero_bubble(self):
        schedules = PipelineScheduleFactory.get_supported_schedules(
            HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
        )
        assert PipelineScheduleType.ZERO_BUBBLE in schedules
        assert PipelineScheduleType.ZBV_ZERO_BUBBLE in schedules
        # Zero-bubble should be first (best)
        assert schedules[0] == PipelineScheduleType.ZERO_BUBBLE

    def test_cuda_blackwell_supports_zero_bubble(self):
        schedules = PipelineScheduleFactory.get_supported_schedules(
            HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC
        )
        assert PipelineScheduleType.ZERO_BUBBLE in schedules

    def test_cuda_ampere_no_zero_bubble(self):
        schedules = PipelineScheduleFactory.get_supported_schedules(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )
        assert PipelineScheduleType.ZERO_BUBBLE not in schedules
        assert PipelineScheduleType.INTERLEAVED_1F1B in schedules

    def test_amd_standard_schedules(self):
        schedules = PipelineScheduleFactory.get_supported_schedules(
            HardwareBackend.AMD, AMDArchitecture.CDNA3
        )
        assert PipelineScheduleType.INTERLEAVED_1F1B in schedules
        assert PipelineScheduleType.GPIPE in schedules
        assert PipelineScheduleType.ZERO_BUBBLE not in schedules

    def test_tpu_limited_schedules(self):
        schedules = PipelineScheduleFactory.get_supported_schedules(
            HardwareBackend.TPU, TPUVersion.V7
        )
        assert PipelineScheduleType.INTERLEAVED_1F1B in schedules
        assert PipelineScheduleType.GPIPE in schedules
        assert PipelineScheduleType.ZERO_BUBBLE not in schedules

    def test_trainium_limited_schedules(self):
        schedules = PipelineScheduleFactory.get_supported_schedules(
            HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2
        )
        assert PipelineScheduleType.INTERLEAVED_1F1B in schedules

    def test_cpu_limited_schedules(self):
        schedules = PipelineScheduleFactory.get_supported_schedules(
            HardwareBackend.CPU
        )
        assert PipelineScheduleType.INTERLEAVED_1F1B in schedules
        assert PipelineScheduleType.ZERO_BUBBLE not in schedules

    def test_optimal_schedule_hopper_2_stages(self):
        schedule = PipelineScheduleFactory.get_optimal_schedule(
            HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER, num_stages=2
        )
        assert schedule == PipelineScheduleType.ZERO_BUBBLE

    def test_optimal_schedule_hopper_4_stages(self):
        schedule = PipelineScheduleFactory.get_optimal_schedule(
            HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER, num_stages=4
        )
        # ZBV requires 4 stages but zero-bubble is first and has min_stages=2
        assert schedule == PipelineScheduleType.ZERO_BUBBLE

    def test_optimal_schedule_ampere(self):
        schedule = PipelineScheduleFactory.get_optimal_schedule(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE, num_stages=2
        )
        assert schedule == PipelineScheduleType.INTERLEAVED_1F1B

    def test_optimal_schedule_cpu(self):
        schedule = PipelineScheduleFactory.get_optimal_schedule(
            HardwareBackend.CPU, num_stages=2
        )
        assert schedule == PipelineScheduleType.INTERLEAVED_1F1B

    def test_get_schedule_spec(self):
        spec = PipelineScheduleFactory.get_schedule_spec(PipelineScheduleType.GPIPE)
        assert isinstance(spec, PipelineScheduleSpec)
        assert spec.schedule == PipelineScheduleType.GPIPE

    def test_torch_schedule_class_names(self):
        assert PipelineScheduleFactory.get_torch_schedule_class_name(
            PipelineScheduleType.GPIPE
        ) == "ScheduleGPipe"
        assert PipelineScheduleFactory.get_torch_schedule_class_name(
            PipelineScheduleType.INTERLEAVED_1F1B
        ) == "ScheduleInterleaved1F1B"
        assert PipelineScheduleFactory.get_torch_schedule_class_name(
            PipelineScheduleType.ZERO_BUBBLE
        ) == "ScheduleInterleavedZeroBubble"
        assert PipelineScheduleFactory.get_torch_schedule_class_name(
            PipelineScheduleType.ZBV_ZERO_BUBBLE
        ) == "ScheduleZBVZeroBubble"
        assert PipelineScheduleFactory.get_torch_schedule_class_name(
            PipelineScheduleType.LOOPED_BFS
        ) == "ScheduleLoopedBFS"
