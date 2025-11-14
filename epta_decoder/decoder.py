import numpy as np
from .utils import gf2_rref, hard_decision_from_beta

class EPTADecoder:
    """Extended Parity-Check Transformation Algorithm Decoder.

    Args:
        H: Parity-check matrix (m x n)
        delta: Update step size (default: 0.05)
        max_iter: Maximum iterations (default: 100)
    """

    def __init__(self, H: np.ndarray, delta: float = 0.05, max_iter: int = 100):
        self.H = H.copy().astype(np.uint8)
        self.m, self.n = self.H.shape
        self.delta = float(delta)
        self.max_iter = int(max_iter)

    def init_beta(self, llr: np.ndarray) -> np.ndarray:
        """Initialize beta probabilities from LLR."""
        p1 = 1.0 / (1.0 + np.exp(-llr))
        beta = np.vstack([1 - p1, p1])
        beta = np.clip(beta, 1e-12, 1.0 - 1e-12)
        return beta

    def transform_parity_check(self, H: np.ndarray, beta: np.ndarray) -> tuple:
        """Transform parity-check matrix to RREF."""
        L = np.max(beta, axis=0)
        l_idx = np.argsort(L)
        H_perm = H[:, l_idx]
        H_rref = gf2_rref(H_perm)
        return H_rref, l_idx

    def parity_check_equations(self, r_hat: np.ndarray, HT: np.ndarray) -> np.ndarray:
        """Compute syndrome."""
        synd = (HT.dot(r_hat) % 2).astype(np.uint8)
        return synd

    def update_beta(self, beta: np.ndarray, HT: np.ndarray, 
                   synd: np.ndarray, l_idx: np.ndarray) -> np.ndarray:
        """Update beta probabilities based on syndrome."""
        delta = self.delta
        for v in range(HT.shape[0]):
            cols = np.where(HT[v, :] == 1)[0]
            if cols.size == 0:
                continue
            if synd[v] == 0:
                beta[:, cols] += delta
            else:
                beta[:, cols] -= delta
        beta = np.clip(beta, 1e-12, 1.0 - 1e-12)
        return beta

    def decode(self, llr: np.ndarray) -> tuple:
        """Decode received LLRs.

        Returns:
            (decoded_bits, info_dict)
        """
        beta = self.init_beta(llr)
        H = self.H.copy()

        for iteration in range(1, self.max_iter + 1):
            HT, l_idx = self.transform_parity_check(H, beta)
            r_hat = hard_decision_from_beta(beta)
            synd = self.parity_check_equations(r_hat, HT)

            if np.all(synd == 0):
                return r_hat, {"success": True, "iterations": iteration}

            beta = self.update_beta(beta, HT, synd, l_idx)

        r_hat = hard_decision_from_beta(beta)
        success = np.all((H.dot(r_hat) % 2) == 0)
        return r_hat, {"success": bool(success), "iterations": self.max_iter}
