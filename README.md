# EPTA Decoder — Extended Parity-Check Transformation Algorithm

Reference implementation of the Extended Parity-Check Transformation Algorithm (EPTA) for iterative soft-decision decoding of binary cyclic codes.

## Features

- Pure Python reference implementation
- Exact (7,3) cyclic code example from paper
- BER simulation and plotting tools
- Unit tests with pytest
- CI/CD with GitHub Actions
- C++/Cython acceleration skeletons
- Packaging for PyPI

## Installation

```bash
# Clone repository
git clone https://github.com/babalolaseyip/epta-decoder.git
cd epta-decoder

# Install package
pip install -e .
```

## Quick Start

```python
import numpy as np
from epta_decoder import EPTADecoder
from epta_decoder.utils import bits_to_bpsk, awgn, llr_from_awgn

# (7,3) parity-check matrix
H = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 0, 1]
], dtype=np.uint8)

# Create decoder
decoder = EPTADecoder(H, delta=0.05, max_iter=100)

# Transmit all-zero codeword
tx = np.zeros(7, dtype=np.uint8)
x = bits_to_bpsk(tx)
y, sigma = awgn(x, snr_db=3.0)
llr = llr_from_awgn(y, sigma)

# Decode
rx, info = decoder.decode(llr)
print(f"Decoded: {rx}, Success: {info['success']}, Iterations: {info['iterations']}")
```

## Running Examples

```bash
# Run (7,3) simulation
python examples/run_7_3.py

# Results saved to examples_data/
# - ber_7_3.npy (data)
# - ber_7_3.png (plot)
```

## Running Tests

```bash
pytest -v
```

## Repository Structure

```
epta-decoder/
├── epta_decoder/          # Core package
│   ├── __init__.py
│   ├── decoder.py         # EPTA decoder
│   └── utils.py           # Utilities
├── examples/              # Example scripts
│   └── run_7_3.py
├── examples_data/         # Output data/plots
├── tests/                 # Unit tests
├── cpp_ext/               # C++ acceleration (skeleton)
├── cython_ext/            # Cython acceleration (skeleton)
└── docs/                  # Documentation
```

## Citation

If you use this software, please cite:

```bibtex
@software{epta_decoder,
  author = {Babalola, Oluwaseyi Paul and Versfeld, Jaco},
  title = {EPTA Decoder},
  year = {2025},
  url = {https://github.com/babalolaseyip/epta-decoder}
}
```

See `CITATION.cff` for full citation metadata.

## License

MIT License - see LICENSE file for details.

## Contributing

See CONTRIBUTING.md for guidelines.
