#!/bin/bash
flake8 ./torch_liberator --count --select=E9,F63,F7,F82 --show-source --statistics
