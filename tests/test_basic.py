import numpy as np
from epta_decoder.decoder import EPTADecoder
from epta_decoder.utils import bits_to_bpsk, awgn, llr_from_awgn


def test_decoder_init():
    """Test decoder initialization."""
    H = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1]
    ], dtype=np.uint8)
    decoder = EPTADecoder(H, delta=0.05, max_iter=20)
    assert decoder.H.shape == (4, 7)
    assert decoder.delta == 0.05
    assert decoder.max_iter == 20


def test_decode_all_zero():
    """Test decoding all-zero codeword at high SNR."""
    H = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1]
    ], dtype=np.uint8)
    decoder = EPTADecoder(H, delta=0.05, max_iter=20)

    # High SNR - should decode correctly
    llr = np.ones(7) * 10.0
    rx, info = decoder.decode(llr)

    assert info['success'] is True
    assert np.array_equal(rx, np.zeros(7, dtype=np.uint8))


def test_utils():
    """Test utility functions."""
    bits = np.array([0, 1, 0, 1])
    bpsk = bits_to_bpsk(bits)
    assert np.array_equal(bpsk, np.array([1, -1, 1, -1]))

    x = np.array([1.0, -1.0])
    y, sigma = awgn(x, snr_db=10)
    assert y.shape == x.shape
    assert sigma > 0
