# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# from omni.isaac.kit import SimulationApp

# simulation_app = SimulationApp({'headless': True})
import os
import sys

from sphinx.builders.html import StandaloneHTMLBuilder

sys.path.insert(0, os.path.abspath('../..'))

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'Intern Robotics Documentation'
copyright = '2025, Intern Robotics'
author = 'Intern Robotics'

# The full version, including alpha/beta/rc tags
release = 'v0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
    'myst_parser',
    'sphinx_copybutton',
]  # yapf: disable

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'pydata_sphinx_theme'
html_logo = '_static/image/logo.png'

# Define the json_url for our version switcher.
json_url = 'https://pydata-sphinx-theme.readthedocs.io/en/latest/_static/switcher.json'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'navbar_start': ['navbar-logo'],
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'navbar_center': ['navbar-nav'],
    'secondary_sidebar_items': ['page-toc', 'edit-this-page', 'sourcelink'],
    'footer_items': ['copyright', 'sphinx-version'],
    'default_mode': 'auto',
    'switcher': {
        'json_url': 'https://pydata-sphinx-theme.readthedocs.io/en/latest/_static/switcher.json',
        'version_match': 'latest',
    },
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/OpenRobotLab/GRUtopia.git',
            'icon': 'fab fa-github-square',
        },
    ],
    'show_nav_level': 2,
    'navigation_depth': 4,
    # 'collapse_navigation': True,
    'show_toc_level': 2,
}

# html_sidebars = {
#     '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']
# }


html_css_files = ['css/readthedocs.css']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Extension configuration -------------------------------------------------
# Ignore >>> when copying code
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True
