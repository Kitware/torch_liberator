#!/bin/bash
rm -rf _skbuild
rm -rf torch_liberator/_nx_ext_v2/*.so
rm -rf torch_liberator/_nx_ext_v2/*.cpp
rm -rf torch_liberator/_nx_ext_v2/*.html

find . -regex ".*\(__pycache__\|\.py[co]\)" -delete || find . -iname "*.pyc" -delete || find . -iname "*.pyo" -delete

rm -rf build
rm -rf wheelhouse
rm -rf dist
