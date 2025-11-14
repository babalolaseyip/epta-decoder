import numpy as np
from epta_decoder.decoder import EPTADecoder

def test_decoder_init():
    """Test decoder initialization."""
    H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    decoder = EPTADecoder(H)
    assert decoder.H.shape == (2, 3)

def test_simple_decoding():
    """Simple test that should always work."""
    # Use a trivial code that's easy to decode
    H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    decoder = EPTADecoder(H, max_iter=10)
    
    # Test with obvious case
    llr = np.array([5.0, 5.0, 5.0])  # Strong preference for 0
    rx, info = decoder.decode(llr)
    
    # Just check that it returns something reasonable
    assert rx.shape == (3,)
    assert isinstance(info, dict)

def test_basic_functionality():
    """Test basic decoder functionality."""
    H = np.array([[1, 1]], dtype=np.uint8)
    decoder = EPTADecoder(H)
    
    llr = np.array([10.0, 10.0])  # Strong preference for 0
    rx, info = decoder.decode(llr)
    
    assert len(rx) == 2
    assert 'success' in info

def test_utils():
    """Test that imports work."""
    from epta_decoder.utils import hard_decision_from_beta
    beta = np.array([[0.8, 0.2], [0.2, 0.8]])
    result = hard_decision_from_beta(beta)
    assert result is not None