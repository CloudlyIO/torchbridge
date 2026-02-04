"""
Tests for the init CLI command.
"""

import os
import tempfile
from unittest.mock import MagicMock

from torchbridge.cli.init import InitCommand


class TestInitCommand:
    """Test the init CLI command."""

    def test_generate_files_training(self):
        """Test generating training template files."""
        files = InitCommand._generate_files("my_project", "training", "auto")
        assert 'train.py' in files
        assert 'config.yaml' in files
        assert 'requirements.txt' in files
        assert 'Dockerfile' in files
        assert '.gitignore' in files
        assert 'README.md' in files
        assert 'serve.py' not in files

    def test_generate_files_inference(self):
        """Test generating inference template files."""
        files = InitCommand._generate_files("my_project", "inference", "auto")
        assert 'serve.py' in files
        assert 'config.yaml' in files
        assert 'requirements.txt' in files
        assert 'Dockerfile' in files
        assert 'train.py' not in files

    def test_generate_files_distributed(self):
        """Test generating distributed template files."""
        files = InitCommand._generate_files("my_project", "distributed", "nvidia")
        assert 'train.py' in files
        assert 'config.yaml' in files
        assert 'requirements.txt' in files
        assert 'NCCL' in files['requirements.txt']
        assert 'distributed' in files['config.yaml']

    def test_generate_files_serving(self):
        """Test generating serving template files."""
        files = InitCommand._generate_files("my_project", "serving", "auto")
        assert 'serve.py' in files
        assert 'config.yaml' in files
        assert 'requirements.txt' in files
        assert 'fastapi' in files['requirements.txt']
        assert 'uvicorn' in files['requirements.txt']
        assert '8000' in files['Dockerfile']

    def test_generated_code_contains_torchbridge_imports(self):
        """Test that generated code contains TorchBridge imports."""
        files = InitCommand._generate_files("test_proj", "training", "auto")
        assert 'torchbridge' in files['train.py']
        assert 'TorchBridgeConfig' in files['train.py']

        files = InitCommand._generate_files("test_proj", "inference", "auto")
        assert 'torchbridge' in files['serve.py']
        assert 'TorchBridgeConfig' in files['serve.py']

    def test_config_contains_backend(self):
        """Test that config files contain the backend setting."""
        files = InitCommand._generate_files("test_proj", "training", "nvidia")
        assert 'nvidia' in files['config.yaml']

        files = InitCommand._generate_files("test_proj", "training", "auto")
        assert 'auto' in files['config.yaml']


class TestInitCommandExecution:
    """Test init command execution with file system."""

    def test_execute_creates_directory(self):
        """Test that execute creates the project directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.name = 'test_project'
            args.template = 'training'
            args.backend = 'auto'
            args.output_dir = tmpdir
            args.force = False
            args.verbose = False

            result = InitCommand.execute(args)
            assert result == 0

            project_dir = os.path.join(tmpdir, 'test_project')
            assert os.path.isdir(project_dir)
            assert os.path.exists(os.path.join(project_dir, 'train.py'))
            assert os.path.exists(os.path.join(project_dir, 'config.yaml'))
            assert os.path.exists(os.path.join(project_dir, 'requirements.txt'))
            assert os.path.exists(os.path.join(project_dir, 'Dockerfile'))
            assert os.path.exists(os.path.join(project_dir, '.gitignore'))
            assert os.path.exists(os.path.join(project_dir, 'README.md'))

    def test_execute_refuses_overwrite_without_force(self):
        """Test that execute refuses to overwrite without --force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, 'existing')
            os.makedirs(project_dir)

            args = MagicMock()
            args.name = 'existing'
            args.template = 'training'
            args.backend = 'auto'
            args.output_dir = tmpdir
            args.force = False
            args.verbose = False

            result = InitCommand.execute(args)
            assert result == 1

    def test_execute_overwrites_with_force(self):
        """Test that execute overwrites with --force."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = os.path.join(tmpdir, 'existing')
            os.makedirs(project_dir)

            args = MagicMock()
            args.name = 'existing'
            args.template = 'training'
            args.backend = 'auto'
            args.output_dir = tmpdir
            args.force = True
            args.verbose = False

            result = InitCommand.execute(args)
            assert result == 0
            assert os.path.exists(os.path.join(project_dir, 'train.py'))

    def test_execute_inference_template(self):
        """Test execute with inference template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.name = 'inference_proj'
            args.template = 'inference'
            args.backend = 'auto'
            args.output_dir = tmpdir
            args.force = False
            args.verbose = False

            result = InitCommand.execute(args)
            assert result == 0

            project_dir = os.path.join(tmpdir, 'inference_proj')
            assert os.path.exists(os.path.join(project_dir, 'serve.py'))
            assert not os.path.exists(os.path.join(project_dir, 'train.py'))

    def test_execute_serving_template(self):
        """Test execute with serving template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.name = 'api_server'
            args.template = 'serving'
            args.backend = 'auto'
            args.output_dir = tmpdir
            args.force = False
            args.verbose = False

            result = InitCommand.execute(args)
            assert result == 0

            project_dir = os.path.join(tmpdir, 'api_server')
            assert os.path.exists(os.path.join(project_dir, 'serve.py'))

            with open(os.path.join(project_dir, 'requirements.txt')) as f:
                reqs = f.read()
            assert 'fastapi' in reqs

    def test_execute_distributed_template(self):
        """Test execute with distributed template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.name = 'dist_proj'
            args.template = 'distributed'
            args.backend = 'nvidia'
            args.output_dir = tmpdir
            args.force = False
            args.verbose = True

            result = InitCommand.execute(args)
            assert result == 0

            project_dir = os.path.join(tmpdir, 'dist_proj')
            assert os.path.exists(os.path.join(project_dir, 'train.py'))

            with open(os.path.join(project_dir, 'config.yaml')) as f:
                config = f.read()
            assert 'distributed' in config

    def test_execute_verbose_output(self, capsys):
        """Test verbose output during execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = MagicMock()
            args.name = 'verbose_proj'
            args.template = 'training'
            args.backend = 'auto'
            args.output_dir = tmpdir
            args.force = False
            args.verbose = True

            result = InitCommand.execute(args)
            assert result == 0

            captured = capsys.readouterr()
            assert 'Created' in captured.out

    def test_readme_content(self):
        """Test README contains quickstart instructions."""
        files = InitCommand._generate_files("my_proj", "training", "auto")
        readme = files['README.md']
        assert 'my_proj' in readme
        assert 'pip install' in readme
        assert 'python train.py' in readme

        files = InitCommand._generate_files("my_proj", "serving", "auto")
        readme = files['README.md']
        assert 'python serve.py' in readme
