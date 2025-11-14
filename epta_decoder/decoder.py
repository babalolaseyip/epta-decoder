import numpy as np
from .utils import gf2_rref, hard_decision_from_beta

class EPTADecoder:
    def __init__(self, H: np.ndarray, delta: float = 0.05, max_iter: int = 100):
        self.H = H.copy().astype(np.uint8)
        self.m, self.n = self.H.shape
        self.delta = float(delta)
        self.max_iter = int(max_iter)

    def init_beta(self, llr: np.ndarray):
        # Convert LLR to probabilities correctly
        # p(x=1) = exp(llr) / (1 + exp(llr))
        # p(x=0) = 1 / (1 + exp(llr))
        p1 = 1.0 / (1.0 + np.exp(-llr))  # P(x=1)
        p0 = 1.0 / (1.0 + np.exp(llr))   # P(x=0)
        
        beta = np.vstack([p0, p1])
        beta = np.clip(beta, 1e-12, 1.0 - 1e-12)
        # Normalize to ensure probabilities sum to 1
        beta = beta / np.sum(beta, axis=0, keepdims=True)
        return beta

    def transform_parity_check(self, H: np.ndarray, beta: np.ndarray):
        # Use reliability measure: max probability
        L = np.max(beta, axis=0)
        l_idx = np.argsort(L)[::-1]  # Sort in descending order (most reliable first)
        H_perm = H[:, l_idx]
        H_rref = gf2_rref(H_perm.copy())
        return H_rref, l_idx

    def update_beta(self, beta: np.ndarray, HT: np.ndarray, synd: np.ndarray, l_idx: np.ndarray):
        delta = self.delta
        beta_updated = beta.copy()
        
        for v in range(HT.shape[0]):
            cols = np.where(HT[v, :] == 1)[0]
            if cols.size == 0:
                continue
                
            if synd[v] == 0:
                # Satisfied parity - increase confidence in current beliefs
                beta_updated[1, cols] += delta  # Increase P(x=1)
            else:
                # Unsatisfied parity - decrease confidence in current beliefs
                beta_updated[0, cols] += delta  # Increase P(x=0)
        
        # Clip and normalize
        beta_updated = np.clip(beta_updated, 1e-12, 1.0 - 1e-12)
        beta_updated = beta_updated / np.sum(beta_updated, axis=0, keepdims=True)
        return beta_updated

    def decode(self, llr: np.ndarray):
        llr = np.asarray(llr, dtype=float)
        if llr.shape[0] != self.n:
            raise ValueError(f"LLR length {llr.shape[0]} doesn't match code length {self.n}")
        
        beta = self.init_beta(llr)
        H = self.H.copy()
        
        for iteration in range(1, self.max_iter + 1):
            HT, l_idx = self.transform_parity_check(H, beta)
            r_hat = hard_decision_from_beta(beta)
            
            if r_hat.ndim > 1:
                r_hat = r_hat.flatten()
            r_hat = r_hat.astype(np.uint8)
            
            # Calculate syndrome
            synd = (HT @ r_hat) % 2
            
            if np.all(synd == 0):
                return r_hat, {"success": True, "iterations": iteration}
            
            beta = self.update_beta(beta, HT, synd, l_idx)
        
        # Final attempt
        r_hat = hard_decision_from_beta(beta).astype(np.uint8).flatten()
        final_synd = (H @ r_hat) % 2
        success = np.all(final_synd == 0)
        
        return r_hat, {"success": bool(success), "iterations": self.max_iter}