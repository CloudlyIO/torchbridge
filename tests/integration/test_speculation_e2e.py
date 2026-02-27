"""
Speculative Decoding End-to-End Integration Tests

Verifies that SpeculationEngine kwargs work correctly with model.generate().
Uses a tiny random-weight GPT2 model (no network download) to test the
full path: engine config → get_generation_kwargs() → model.generate().

Skipped automatically if 'transformers' is not installed.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from torchbridge.core.config import HardwareBackend  # noqa: E402
from torchbridge.inference.speculative import (  # noqa: E402
    SpeculationConfig,
    SpeculationEngine,
)
from torchbridge.inference.speculative.methods import SpeculativeMethod  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_gpt2() -> GPT2LMHeadModel:
    """Random-weight GPT2 with minimal configuration — no download required.

    Weights are seeded for reproducibility. The module scope avoids recreating
    the model for every test (no state leaks — generate() resets KV cache per call
    and the model is in eval() mode with no_grad).
    """
    torch.manual_seed(42)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=200)
    model = GPT2LMHeadModel(cfg)
    model.eval()
    return model


@pytest.fixture
def prompt_lookup_engine() -> SpeculationEngine:
    return SpeculationEngine(
        config=SpeculationConfig(
            method=SpeculativeMethod.PROMPT_LOOKUP,
            num_speculative_tokens=3,
        ),
        backend=HardwareBackend.CPU,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPromptLookupKwargsInGenerate:
    """PROMPT_LOOKUP kwargs must be accepted by and work with model.generate()."""

    def test_kwargs_accepted_by_generate(
        self, tiny_gpt2: GPT2LMHeadModel, prompt_lookup_engine: SpeculationEngine
    ) -> None:
        """model.generate() must not raise when passed prompt_lookup_num_tokens."""
        kwargs = prompt_lookup_engine.get_generation_kwargs()
        input_ids = torch.randint(0, 200, (1, 10))
        # Should not raise
        output = tiny_gpt2.generate(input_ids, max_new_tokens=3, **kwargs)
        assert output is not None

    def test_output_is_longer_than_input(
        self, tiny_gpt2: GPT2LMHeadModel, prompt_lookup_engine: SpeculationEngine
    ) -> None:
        """Output sequence must be longer than the input after generation."""
        kwargs = prompt_lookup_engine.get_generation_kwargs()
        input_ids = torch.randint(0, 200, (1, 10))
        output = tiny_gpt2.generate(input_ids, max_new_tokens=3, **kwargs)
        assert output.shape[1] > input_ids.shape[1]

    def test_auto_resolved_cpu_engine_kwargs_work(
        self, tiny_gpt2: GPT2LMHeadModel
    ) -> None:
        """Auto-resolved CPU engine should produce PROMPT_LOOKUP kwargs that work."""
        engine = SpeculationEngine(backend=HardwareBackend.CPU)
        assert engine.method == SpeculativeMethod.PROMPT_LOOKUP
        kwargs = engine.get_generation_kwargs()
        assert "prompt_lookup_num_tokens" in kwargs
        input_ids = torch.randint(0, 200, (1, 10))
        output = tiny_gpt2.generate(input_ids, max_new_tokens=2, **kwargs)
        assert output.shape[1] > input_ids.shape[1]

    def test_different_num_speculative_tokens(
        self, tiny_gpt2: GPT2LMHeadModel
    ) -> None:
        """Various num_speculative_tokens values should all work in generate()."""
        input_ids = torch.randint(0, 200, (1, 12))
        for n in (1, 3, 5):
            engine = SpeculationEngine(
                config=SpeculationConfig(
                    method=SpeculativeMethod.PROMPT_LOOKUP,
                    num_speculative_tokens=n,
                ),
                backend=HardwareBackend.CPU,
            )
            kwargs = engine.get_generation_kwargs()
            assert kwargs["prompt_lookup_num_tokens"] == n
            output = tiny_gpt2.generate(input_ids, max_new_tokens=2, **kwargs)
            assert output.shape[1] > input_ids.shape[1]

    def test_disabled_engine_empty_kwargs_work_in_generate(
        self, tiny_gpt2: GPT2LMHeadModel
    ) -> None:
        """Disabled engine returns {} — model.generate() must still work."""
        engine = SpeculationEngine(
            config=SpeculationConfig(enabled=False),
            backend=HardwareBackend.CPU,
        )
        kwargs = engine.get_generation_kwargs()
        assert kwargs == {}
        input_ids = torch.randint(0, 200, (1, 8))
        output = tiny_gpt2.generate(input_ids, max_new_tokens=2, **kwargs)
        assert output.shape[1] > input_ids.shape[1]

    def test_kwargs_consistent_across_calls(
        self, prompt_lookup_engine: SpeculationEngine
    ) -> None:
        """Repeated get_generation_kwargs() calls return identical dicts."""
        first = prompt_lookup_engine.get_generation_kwargs()
        second = prompt_lookup_engine.get_generation_kwargs()
        assert first == second

    def test_prompt_lookup_larger_than_seq_does_not_crash(
        self, tiny_gpt2: GPT2LMHeadModel
    ) -> None:
        """prompt_lookup_num_tokens > sequence length must not crash generate()."""
        engine = SpeculationEngine(
            config=SpeculationConfig(
                method=SpeculativeMethod.PROMPT_LOOKUP,
                num_speculative_tokens=50,  # larger than input
            ),
            backend=HardwareBackend.CPU,
        )
        kwargs = engine.get_generation_kwargs()
        input_ids = torch.randint(0, 200, (1, 5))  # short input
        # HuggingFace clips internally; must not raise
        output = tiny_gpt2.generate(input_ids, max_new_tokens=2, **kwargs)
        assert output.shape[1] > input_ids.shape[1]

    def test_correct_kwarg_name_and_value(self, prompt_lookup_engine: SpeculationEngine) -> None:
        """Kwarg key is 'prompt_lookup_num_tokens' and value matches config."""
        kwargs = prompt_lookup_engine.get_generation_kwargs()
        assert set(kwargs.keys()) == {"prompt_lookup_num_tokens"}
        assert kwargs["prompt_lookup_num_tokens"] == 3
        assert "assistant_model" not in kwargs
