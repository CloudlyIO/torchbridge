"""
TorchBridge Serving — removed in v0.5.56.

Serving infrastructure (LLM server, FastAPI, TorchServe, Triton) violated
Rule 0 (doesn't validate or configure across backends) and Rule 4 (competes
with vLLM, Ray Serve, TorchServe, and the Triton SDK).

For model serving use: vLLM, Ray Serve, FastAPI, or TorchServe directly.
For cross-backend validation use: tb-validate --compare
"""

__all__: list[str] = []
