#!/usr/bin/env python3
"""
(7,3) Cyclic Code EPTA Simulation

Reproduces BER performance using exact parity-check matrix from paper.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from epta_decoder.utils import bits_to_bpsk, awgn, llr_from_awgn
from epta_decoder.decoder import EPTADecoder


def run_simulation(trials=200, snrs=None):
    """Run BER simulation for (7,3) code."""
    if snrs is None:
        snrs = [0, 1, 2, 3, 4, 5]

    # Parity-check matrix from paper (systematic form)
    H = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1]
    ], dtype=np.uint8)

    decoder = EPTADecoder(H, delta=0.05, max_iter=100)
    n = H.shape[1]

    print(f"Running simulation with {trials} trials per SNR...")
    print(f"Code: (7,3) cyclic code")
    print(f"SNR range: {snrs} dB")
    print("-" * 50)

    bers = []
    for snr in snrs:
        errs = 0
        total_bits = 0

        for t in range(trials):
            # All-zero codeword for simplicity
            tx = np.zeros(n, dtype=np.uint8)
            x = bits_to_bpsk(tx)
            y, sigma = awgn(x, snr)
            llr = llr_from_awgn(y, sigma)
            rx, info = decoder.decode(llr)
            errs += np.sum(rx != tx)
            total_bits += n

        ber_val = errs / total_bits
        bers.append(ber_val)
        print(f"SNR {snr:2d} dB => BER {ber_val:.3e}")

    # Save results
    outdir = os.path.join(os.path.dirname(__file__), '..', 'examples_data')
    os.makedirs(outdir, exist_ok=True)

    data = np.array([snrs, bers])
    np.save(os.path.join(outdir, 'ber_7_3.npy'), data)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(snrs, bers, marker='o', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('(7,3) Cyclic Code - EPTA Decoder Performance', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'ber_7_3.png'), dpi=150)

    print("-" * 50)
    print(f"Results saved to {outdir}/")
    print("  - ber_7_3.npy (data)")
    print("  - ber_7_3.png (plot)")

    return snrs, bers


if __name__ == '__main__':
    run_simulation()
