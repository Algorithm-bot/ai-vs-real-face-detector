"""Frequency and upsampling artifact branch for AI-generated image detection."""

from src.freq_branch.extractor import FREQ_FEATURE_DIM, FreqExtractor, FreqResult

__all__ = ["FreqExtractor", "FreqResult", "FREQ_FEATURE_DIM"]
