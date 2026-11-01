"""Tests for branch-only training modes to verify correct feature extractor usage."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestBranchOnlyModes:
    """Test that branch-only modes only use the correct feature extractors."""

    def test_physics_only_mode_does_not_call_prnu_extractor(self):
        """
        Verify that in physics_only mode, PRNU extractor is never initialized or called.
        """
        from src.features.feature_scalers import fit_physics_scaler
        from src.prnu_branch.extractor import PRNUExtractor

        # Create mock image paths
        mock_paths = [f"fake_path_{i}.jpg" for i in range(5)]

        # Mock the PRNUExtractor to track if it's ever instantiated
        with patch('src.features.feature_scalers.PRNUExtractor', autospec=True) as mock_prnu_class:
            mock_prnu_instance = MagicMock()
            mock_prnu_class.return_value = mock_prnu_instance

            # Mock the PhysicsFeatureExtractor
            with patch('src.features.feature_scalers.PhysicsFeatureExtractor') as mock_physics_class:
                mock_physics_instance = MagicMock()
                mock_physics_class.return_value = mock_physics_instance
                mock_physics_instance.extract.return_value.vector = np.zeros(22, dtype=np.float32)
                mock_physics_instance.close = MagicMock()

                # Mock image loading
                with patch('src.features.feature_scalers._load_rgb') as mock_load_rgb:
                    mock_load_rgb.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

                    # Call fit_physics_scaler (used in physics_only mode)
                    fit_physics_scaler(mock_paths, max_samples=5, seed=42)

                    # Verify PRNUExtractor was NEVER instantiated
                    mock_prnu_class.assert_not_called()

                    # Verify PhysicsFeatureExtractor WAS instantiated and used
                    mock_physics_class.assert_called_once()
                    assert mock_physics_instance.extract.call_count == 5

    def test_prnu_only_mode_does_not_call_physics_extractor(self):
        """
        Verify that in prnu_only mode, Physics extractor is never initialized or called.
        """
        from src.features.feature_scalers import fit_prnu_scaler
        from src.physics_branch.feature_vector import PhysicsFeatureExtractor

        # Create mock image paths
        mock_paths = [f"fake_path_{i}.jpg" for i in range(5)]

        # Mock the PhysicsFeatureExtractor to track if it's ever instantiated
        with patch('src.features.feature_scalers.PhysicsFeatureExtractor', autospec=True) as mock_physics_class:
            mock_physics_instance = MagicMock()
            mock_physics_class.return_value = mock_physics_instance

            # Mock the PRNUExtractor
            with patch('src.features.feature_scalers.PRNUExtractor') as mock_prnu_class:
                mock_prnu_instance = MagicMock()
                mock_prnu_class.return_value = mock_prnu_instance
                mock_prnu_instance.extract.return_value.vector = np.zeros(9, dtype=np.float32)

                # Mock image loading
                with patch('src.features.feature_scalers._load_rgb') as mock_load_rgb:
                    mock_load_rgb.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

                    # Call fit_prnu_scaler (used in prnu_only mode)
                    fit_prnu_scaler(mock_paths, max_samples=5, seed=42)

                    # Verify PhysicsFeatureExtractor was NEVER instantiated
                    mock_physics_class.assert_not_called()

                    # Verify PRNUExtractor WAS instantiated and used
                    mock_prnu_class.assert_called_once()
                    assert mock_prnu_instance.extract.call_count == 5

    def test_full_hybrid_mode_calls_both_extractors(self):
        """
        Verify that in full_hybrid mode, both Physics and PRNU extractors are called.
        """
        from src.features.feature_scalers import fit_physics_and_prnu_scalers

        # Create mock image paths
        mock_paths = [f"fake_path_{i}.jpg" for i in range(5)]

        # Mock both extractors
        with patch('src.features.feature_scalers.PhysicsFeatureExtractor') as mock_physics_class, \
             patch('src.features.feature_scalers.PRNUExtractor') as mock_prnu_class, \
             patch('src.features.feature_scalers._load_rgb') as mock_load_rgb:

            mock_physics_instance = MagicMock()
            mock_physics_class.return_value = mock_physics_instance
            mock_physics_instance.extract.return_value.vector = np.zeros(22, dtype=np.float32)
            mock_physics_instance.close = MagicMock()

            mock_prnu_instance = MagicMock()
            mock_prnu_class.return_value = mock_prnu_instance
            mock_prnu_instance.extract.return_value.vector = np.zeros(9, dtype=np.float32)

            mock_load_rgb.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

            # Call fit_physics_and_prnu_scalers (used in full_hybrid mode)
            fit_physics_and_prnu_scalers(mock_paths, max_samples=5, seed=42)

            # Verify BOTH extractors were instantiated and used
            mock_physics_class.assert_called_once()
            mock_prnu_class.assert_called_once()
            assert mock_physics_instance.extract.call_count == 5
            assert mock_prnu_instance.extract.call_count == 5

    def test_physics_only_scaler_logs_correct_message(self, capsys):
        """
        Verify that fit_physics_scaler logs the correct progress message.
        """
        from src.features.feature_scalers import fit_physics_scaler

        mock_paths = [f"fake_path_{i}.jpg" for i in range(3)]

        with patch('src.features.feature_scalers.PhysicsFeatureExtractor') as mock_physics_class, \
             patch('src.features.feature_scalers._load_rgb') as mock_load_rgb:

            mock_physics_instance = MagicMock()
            mock_physics_class.return_value = mock_physics_instance
            mock_physics_instance.extract.return_value.vector = np.zeros(22, dtype=np.float32)
            mock_physics_instance.close = MagicMock()
            mock_load_rgb.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

            fit_physics_scaler(mock_paths, max_samples=3, seed=42)

            # The progress bar is from tqdm, so we check the extractor was called correctly
            assert mock_physics_instance.extract.call_count == 3

    def test_prnu_only_scaler_logs_correct_message(self, capsys):
        """
        Verify that fit_prnu_scaler logs the correct progress message.
        """
        from src.features.feature_scalers import fit_prnu_scaler

        mock_paths = [f"fake_path_{i}.jpg" for i in range(3)]

        with patch('src.features.feature_scalers.PRNUExtractor') as mock_prnu_class, \
             patch('src.features.feature_scalers._load_rgb') as mock_load_rgb:

            mock_prnu_instance = MagicMock()
            mock_prnu_class.return_value = mock_prnu_instance
            mock_prnu_instance.extract.return_value.vector = np.zeros(9, dtype=np.float32)
            mock_load_rgb.return_value = np.zeros((224, 224, 3), dtype=np.uint8)

            fit_prnu_scaler(mock_paths, max_samples=3, seed=42)

            # The progress bar is from tqdm, so we check the extractor was called correctly
            assert mock_prnu_instance.extract.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])