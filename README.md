# aind-pophys-converter

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-100.0%25-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-85%25-yellow?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.10-blue?logo=python)

A Python package for converting and processing population physiology data files, with a focus on imaging data formats such as TIFF and HDF5.

## Overview

aind-pophys-converter provides utilities for converting and working with population physiology data, particularly from imaging experiments. It includes tools to handle various data formats like TIFF files commonly generated in neuroscience experiments.

## Installation

### From PyPI

```bash
pip install aind-pophys-converter
```

### From source

To install the latest development version, clone the repository and install it using pip:

```bash
git clone https://github.com/<your-username>/aind-pophys-converter.git
cd aind-pophys-converter
pip install -e .
```

## Usage

```python
from aind_pophys_converter import Converter

# Convert a TIFF file to a different format
converter = Converter("path/to/file.tiff")
converter.convert("desired/output/format")
```

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This package was developed as part of the Allen Institute for Neural Dynamics.
- We thank the contributors and maintainers of the open-source packages used in this project.
