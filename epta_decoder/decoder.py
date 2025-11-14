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
        llr = np.clip(llr, -20, 20)  # Prevent overflow
        p1 = 1.0 / (1.0 + np.exp(-llr))
        beta = np.vstack([1 - p1, p1])
        beta = np.clip(beta, 1e-12, 1.0 - 1e-12)
        # Normalize
        beta = beta / np.sum(beta, axis=0, keepdims=True)
        return beta
    
    def transform_parity_check(self, H: np.ndarray, beta: np.ndarray) -> tuple:
        """Transform parity-check matrix to RREF."""
        # Use reliability (max probability) for sorting
        L = np.max(beta, axis=0)
        l_idx = np.argsort(L)  # Sort ascending (least reliable first)
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
        beta_new = beta.copy()
        
        for v in range(HT.shape[0]):
            # Find columns (permuted indices) in this parity check
            cols = np.where(HT[v, :] == 1)[0]
            if cols.size == 0:
                continue
            
            # Map back to original indices using l_idx
            orig_cols = l_idx[cols]
            
            if synd[v] == 0:
                # Syndrome satisfied: increase probability of current values
                for col in orig_cols:
                    current_bit = np.argmax(beta[:, col])
                    beta_new[current_bit, col] += delta
            else:
                # Syndrome not satisfied: decrease probability of current values
                for col in orig_cols:
                    current_bit = np.argmax(beta[:, col])
                    beta_new[current_bit, col] -= delta
        
        # Ensure valid probabilities
        beta_new = np.clip(beta_new, 1e-12, 1.0 - 1e-12)
        # Normalize to sum to 1
        beta_new = beta_new / np.sum(beta_new, axis=0, keepdims=True)
        
        return beta_new
    
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
            
            # Check syndrome on original H matrix
            synd_original = (H.dot(r_hat) % 2).astype(np.uint8)
            if np.all(synd_original == 0):
                return r_hat, {"success": True, "iterations": iteration}
            
            # Compute syndrome on transformed matrix for updates
            synd = self.parity_check_equations(r_hat, HT)
            beta = self.update_beta(beta, HT, synd, l_idx)
        
        r_hat = hard_decision_from_beta(beta)
        success = np.all((H.dot(r_hat) % 2) == 0)
        return r_hat, {"success": bool(success), "iterations": self.max_iter}