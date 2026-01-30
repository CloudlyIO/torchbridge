"""Tests for the kpt-profile CLI command."""

import json

import pytest
import torch

from kernel_pytorch.cli.profile import ProfileCommand


class TestProfileCommand:
    """Tests for the ProfileCommand class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )

    @pytest.fixture
    def model_path(self, simple_model, tmp_path):
        """Save model and return path."""
        path = tmp_path / "test_model.pt"
        torch.save(simple_model, path)
        return path

    def test_register_adds_subparser(self):
        """Test that register adds the profile subparser."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        ProfileCommand.register(subparsers)

        # Parse with profile command
        args = parser.parse_args(['profile', '--model', 'test.pt'])
        assert args.model == 'test.pt'

    def test_profile_summary_mode(self, model_path, tmp_path):
        """Test profiling in summary mode."""
        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            mode='summary',
            input_shape='1,512',
            dtype='float32',
            device='cpu',
            iterations=5,
            warmup=2,
            output=None,
            sort_by='cpu_time',
            top_n=10,
            with_stack=False,
            with_modules=False,
            verbose=False,
            quiet=True,
        )

        result = ProfileCommand.execute(args)

        assert result == 0

    def test_profile_detailed_mode(self, model_path, tmp_path):
        """Test profiling in detailed mode."""
        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            mode='detailed',
            input_shape='1,512',
            dtype='float32',
            device='cpu',
            iterations=3,
            warmup=1,
            output=None,
            sort_by='cpu_time',
            top_n=10,
            with_stack=False,
            with_modules=False,
            verbose=False,
            quiet=True,
        )

        result = ProfileCommand.execute(args)

        assert result == 0

    def test_profile_memory_mode(self, model_path, tmp_path):
        """Test profiling in memory mode."""
        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            mode='memory',
            input_shape='1,512',
            dtype='float32',
            device='cpu',
            iterations=3,
            warmup=1,
            output=None,
            sort_by='cpu_time',
            top_n=10,
            with_stack=False,
            with_modules=False,
            verbose=False,
            quiet=True,
        )

        result = ProfileCommand.execute(args)

        assert result == 0

    def test_profile_trace_mode(self, model_path, tmp_path):
        """Test profiling in trace mode."""
        trace_output = tmp_path / "trace.json"

        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            mode='trace',
            input_shape='1,512',
            dtype='float32',
            device='cpu',
            iterations=3,
            warmup=1,
            output=str(trace_output),
            sort_by='cpu_time',
            top_n=10,
            with_stack=False,
            with_modules=False,
            verbose=False,
            quiet=True,
        )

        result = ProfileCommand.execute(args)

        assert result == 0
        assert trace_output.exists()

    def test_profile_with_json_output(self, model_path, tmp_path):
        """Test profiling with JSON output."""
        json_output = tmp_path / "profile.json"

        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            mode='summary',
            input_shape='1,512',
            dtype='float32',
            device='cpu',
            iterations=3,
            warmup=1,
            output=str(json_output),
            sort_by='cpu_time',
            top_n=10,
            with_stack=False,
            with_modules=False,
            verbose=False,
            quiet=True,
        )

        result = ProfileCommand.execute(args)

        assert result == 0
        assert json_output.exists()

        # Verify JSON is valid
        with open(json_output) as f:
            data = json.load(f)

        assert 'latency' in data
        assert 'model' in data
        assert 'config' in data

    def test_profile_with_fp16(self, model_path, tmp_path):
        """Test profiling with FP16 dtype."""
        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            mode='summary',
            input_shape='1,512',
            dtype='float16',
            device='cpu',
            iterations=3,
            warmup=1,
            output=None,
            sort_by='cpu_time',
            top_n=10,
            with_stack=False,
            with_modules=False,
            verbose=False,
            quiet=True,
        )

        result = ProfileCommand.execute(args)

        assert result == 0

    def test_profile_auto_device(self, model_path, tmp_path):
        """Test profiling with auto device detection."""
        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            mode='summary',
            input_shape='1,512',
            dtype='float32',
            device='auto',
            iterations=3,
            warmup=1,
            output=None,
            sort_by='cpu_time',
            top_n=10,
            with_stack=False,
            with_modules=False,
            verbose=False,
            quiet=True,
        )

        result = ProfileCommand.execute(args)

        assert result == 0

    def test_parse_shape(self):
        """Test shape parsing."""
        assert ProfileCommand._parse_shape("1,512") == (1, 512)
        assert ProfileCommand._parse_shape("2,3,224,224") == (2, 3, 224, 224)
        assert ProfileCommand._parse_shape("1") == (1,)

    def test_profile_different_input_shapes(self, model_path, tmp_path):
        """Test profiling with different input shapes."""
        for shape in ["1,512", "4,512", "8,512"]:
            import argparse
            args = argparse.Namespace(
                model=str(model_path),
                mode='summary',
                input_shape=shape,
                dtype='float32',
                device='cpu',
                iterations=3,
                warmup=1,
                output=None,
                sort_by='cpu_time',
                top_n=10,
                with_stack=False,
                with_modules=False,
                verbose=False,
                quiet=True,
            )

            result = ProfileCommand.execute(args)

            assert result == 0

    def test_profile_nonexistent_model_fallback(self, tmp_path):
        """Test that nonexistent model falls back to simple model."""
        import argparse
        args = argparse.Namespace(
            model='nonexistent_model',
            mode='summary',
            input_shape='1,512',
            dtype='float32',
            device='cpu',
            iterations=3,
            warmup=1,
            output=None,
            sort_by='cpu_time',
            top_n=10,
            with_stack=False,
            with_modules=False,
            verbose=False,
            quiet=True,
        )

        # Should not raise, falls back to simple model
        result = ProfileCommand.execute(args)
        assert result == 0


class TestProfileDetailedOptions:
    """Tests for detailed profiling options."""

    @pytest.fixture
    def simple_model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )

    @pytest.fixture
    def model_path(self, simple_model, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(simple_model, path)
        return path

    def test_detailed_with_stack_traces(self, model_path, tmp_path):
        """Test detailed profiling with stack traces."""
        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            mode='detailed',
            input_shape='1,512',
            dtype='float32',
            device='cpu',
            iterations=3,
            warmup=1,
            output=None,
            sort_by='cpu_time',
            top_n=10,
            with_stack=True,
            with_modules=False,
            verbose=False,
            quiet=True,
        )

        result = ProfileCommand.execute(args)

        assert result == 0

    def test_detailed_with_modules(self, model_path, tmp_path):
        """Test detailed profiling with module grouping."""
        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            mode='detailed',
            input_shape='1,512',
            dtype='float32',
            device='cpu',
            iterations=3,
            warmup=1,
            output=None,
            sort_by='cpu_time',
            top_n=10,
            with_stack=False,
            with_modules=True,
            verbose=False,
            quiet=True,
        )

        result = ProfileCommand.execute(args)

        assert result == 0

    def test_detailed_sort_by_options(self, model_path, tmp_path):
        """Test detailed profiling with different sort options."""
        for sort_by in ['cpu_time', 'count']:
            import argparse
            args = argparse.Namespace(
                model=str(model_path),
                mode='detailed',
                input_shape='1,512',
                dtype='float32',
                device='cpu',
                iterations=3,
                warmup=1,
                output=None,
                sort_by=sort_by,
                top_n=10,
                with_stack=False,
                with_modules=False,
                verbose=False,
                quiet=True,
            )

            result = ProfileCommand.execute(args)

            assert result == 0
