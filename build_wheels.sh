#!/bin/bash
__doc__="""
SeeAlso:
    pyproject.toml
"""
#pip wheel -w wheelhouse .
# python -m build --wheel -o wheelhouse  #  torch_liberator: +COMMENT_IF(binpy)
cibuildwheel --config-file pyproject.toml --platform linux --arch x86_64  #  torch_liberator: +UNCOMMENT_IF(binpy)
