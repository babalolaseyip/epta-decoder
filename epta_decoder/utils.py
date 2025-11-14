import numpy as np

def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    """Convert binary bits to BPSK symbols (0->1, 1->-1)."""
    return 1.0 - 2.0 * bits

def awgn(x: np.ndarray, snr_db: float) -> tuple:
    """Add AWGN noise to signal.

    Returns:
        (noisy_signal, noise_std)
    """
    snr = 10 ** (snr_db / 10.0)
    p_signal = np.mean(x ** 2)
    sigma2 = p_signal / (2 * snr + 1e-12)
    noise = np.sqrt(sigma2) * np.random.randn(*x.shape)
    return x + noise, np.sqrt(sigma2)

def llr_from_awgn(y: np.ndarray, sigma: float) -> np.ndarray:
    """Compute log-likelihood ratios from AWGN channel output."""
    return 2.0 * y / (sigma ** 2 + 1e-12)

def ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Calculate bit error rate."""
    return float(np.mean(tx_bits != rx_bits))

def hard_decision_from_beta(beta: np.ndarray) -> np.ndarray:
    """Make hard decision from beta probabilities."""
    return np.argmax(beta, axis=0).astype(np.uint8)

def gf2_rref(A: np.ndarray) -> np.ndarray:
    """Compute reduced row echelon form over GF(2)."""
    A = A.copy() % 2
    m, n = A.shape
    r = 0
    for c in range(n):
        if r >= m:
            break
        # Find pivot
        piv = None
        for i in range(r, m):
            if A[i, c] == 1:
                piv = i
                break
        if piv is None:
            continue
        # Swap rows
        if piv != r:
            A[[r, piv], :] = A[[piv, r], :]
        # Eliminate
        for i in range(m):
            if i != r and A[i, c] == 1:
                A[i, :] = (A[i, :] + A[r, :]) % 2
        r += 1
    return A
