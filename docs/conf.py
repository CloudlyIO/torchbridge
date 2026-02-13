# TorchBridge Documentation Configuration
# Sphinx configuration file

import os
import sys

# Add source directory to path for autodoc
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

# -- Project information --
project = "TorchBridge"
author = "TorchBridge Team"
release = "0.5.21"

# -- General configuration --
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Support both .rst and .md
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output --
html_theme = "furo"
html_static_path = ["_static"]
html_title = "TorchBridge Documentation"

# -- Autodoc configuration --
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# -- Autosummary --
autosummary_generate = True

# -- Napoleon (Google/NumPy docstrings) --
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Intersphinx --
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- MyST (Markdown) --
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
