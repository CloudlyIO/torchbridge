TorchBridge Documentation
=========================

**TorchBridge** validates that your model produces correct outputs across PyTorch backends
and recommends optimal hardware configurations.
Validate once. Trust everywhere.

.. code-block:: bash

   pip install torchbridge-ml

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guides/backend-selection
   guides/cli
   guides/deployment
   guides/distributed-training
   guides/hardware-setup
   guides/performance-tuning
   guides/data-handling
   guides/use-cases

.. toctree::
   :maxdepth: 2
   :caption: Backend Reference

   backends/overview
   backends/nvidia
   backends/amd
   backends/tpu
   backends/trainium

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/hardware-matrix
   reference/compatibility-matrix
   reference/cloud-validation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/backends
   api/cli
   api/precision

API Reference
-------------

.. autosummary::
   :toctree: _autosummary
   :recursive:

   torchbridge.core
   torchbridge.backends
   torchbridge.cli
   torchbridge.precision

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
