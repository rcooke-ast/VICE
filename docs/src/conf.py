# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
try: 
	ModuleNotFoundError 
except NameError: 
	ModuleNotFoundError = ImportError 
try: 
	import sphinx 
except (ModuleNotFoundError, ImportError): 
	raise RuntimeError("""Sphinx >= 2.0.0 is required to compile VICE's \
documentation.""") 
version_info_from_string = lambda s: tuple([int(i) for i in s.split('.')]) 
if version_info_from_string(sphinx.__version__)[:2] < (2, 0): 
	raise RuntimeError("Must have Sphinx version >= 2.0.0. Current: %s" % (
		sphinx.__version__)) 
else: pass 
import warnings 
warnings.filterwarnings("ignore") 
try: 
	import vice 
except (ModuleNotFoundError, ImportError): 
	raise RuntimeError("""VICE not found. VICE must be installed before the \
documentation can be compiled.""") 




# -- Project information -----------------------------------------------------

project = 'VICE'
copyright = '2020, James W. Johnson'
author = 'James W. Johnson'

# The full version, including alpha/beta/rc tags
release = '1.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
]

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
# html_theme = "classic" 
html_theme = "nature" 
# html_theme = "sphinx_rtd_theme" 

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# html_theme_options = {
# 	"linkcolor": 			"blue", 
# 	"visitedlinkcolor": 	"blue"  
# }

latex_elements = {
	# "tableofcontents": 	r"\tableofcontents" 
}

