# !/bin/bash

set -e

# perform lint test for all files.
pylint -r n --disable=fixme --output-format=colorized *.py tests/python

# perform nosetests for common libraries.
echo "Performing nosetests."
nosetests
