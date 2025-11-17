# Configuration file for the Sphinx documentation builder

import os
import sys
from datetime import datetime

# ──────────── Path setup ────────────
# Add project root so autodoc can import the Python package
sys.path.insert(0, os.path.abspath(".."))

# ──────────── Project information ────────────
project = "volumembo"
author = "Fabius Krämer, Thomas Isensee"
year = datetime.now().year
copyright = f"{year}, {author}"
release = "0.1.0"

# ──────────── Extensions ────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.intersphinx",
    "breathe",
    "exhale",
]

# Where reStructuredText lives
templates_path = ["_templates"]

# ──────────── HTML output ────────────
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# ──────────── Autodoc settings ────────────
autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_typehints = "description"

# ──────────── Intersphinx (optional but nice) ────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# ──────────── Breathe configuration ────────────
breathe_projects = {"volumembo": "./_doxygen/xml"}
breathe_default_project = "volumembo"

# ──────────── Exhale configuration ────────────
exhale_args = {
    # These arguments are required
    "containmentFolder": "./cppapi",
    "rootFileName": "cpp_api_root.rst",
    "doxygenStripFromPath": "..",
    # Heavily encouraged optional argument
    "rootFileTitle": "Library API",
    # Suggested optional arguments
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin": "INPUT = ../include",
}

# Tell sphinx what the primary language being documented is
primary_domain = "cpp"

# Tell sphinx what the pygments highlight language should be
highlight_language = "cpp"
