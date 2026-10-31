from regimes.detector import RegimeDetector


def test_regime_detector_returns_regime_information(sample_ohlcv):
    detector = RegimeDetector()

    result = detector.latest_regime(sample_ohlcv)

    assert isinstance(result, dict)
    assert "regime" in result
    assert isinstance(result["regime"], str)