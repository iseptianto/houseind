import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.api.schemas import OLXPredictionRequest
from src.api.inference import predict_price, _engineer_features, _to_row

class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_engineer_features_basic(self):
        """Test basic feature engineering."""
        df = pd.DataFrame({
            'LB': [120.0],
            'LT': [150.0],
            'KM': [2],
            'KT': [3],
            'Kota/Kab': ['Jakarta Selatan'],
            'Provinsi': ['Jakarta D.K.I.'],
            'Type': ['rumah']
        })

        result = _engineer_features(df)

        # Check that required columns are present
        expected_cols = ['LB', 'LT', 'KM', 'KT', 'Kota/Kab', 'Provinsi', 'Type',
                        'LBxLT', 'log_LB', 'log_LT', 'lb_x_km', 'lt_x_kt', 'ratio_lb_lt']
        for col in expected_cols:
            assert col in result.columns

        # Check calculated values
        assert result['LBxLT'].iloc[0] == 120.0 * 150.0
        assert result['log_LB'].iloc[0] == np.log1p(120.0)
        assert result['lb_x_km'].iloc[0] == 120.0 * 2

    def test_to_row_conversion(self):
        """Test request to row conversion."""
        req = OLXPredictionRequest(
            LB=120.0,
            LT=150.0,
            KM=2,
            KT=3,
            kota_kab="Jakarta Selatan",
            provinsi="Jakarta D.K.I.",
            type_="rumah"
        )

        result = _to_row(req)

        expected = {
            'LB': 120.0,
            'LT': 150.0,
            'KM': 2,
            'KT': 3,
            'Kota/Kab': 'Jakarta Selatan',
            'Provinsi': 'Jakarta D.K.I.',
            'Type': 'rumah',
            'harga_per_m2': None,
            'ratio_bangunan_rumah': None
        }

        assert result == expected

class TestPrediction:
    """Test prediction functionality."""

    @patch('src.api.inference._ensure_loaded')
    @patch('src.api.inference._preproc')
    @patch('src.api.inference._model')
    def test_predict_price_success(self, mock_model, mock_preproc, mock_ensure_loaded):
        """Test successful price prediction."""
        # Setup mocks
        mock_preproc.transform.return_value = np.array([[1.0, 2.0, 3.0]])
        mock_model.predict.return_value = np.array([500000000.0])
        mock_model.__class__.__name__ = 'GradientBoostingRegressor'

        # Create test request
        req = OLXPredictionRequest(
            LB=120.0,
            LT=150.0,
            KM=2,
            KT=3,
            kota_kab="Jakarta Selatan",
            provinsi="Jakarta D.K.I.",
            type_="rumah"
        )

        result = predict_price(req)

        assert result.prediction == 500000000.0
        assert result.confidence_score == 0.85  # Default for regression
        assert result.model_name == 'GradientBoostingRegressor'
        assert len(result.price_range) == 2
        assert 'prediction_time' in result.__dict__

    @patch('src.api.inference._ensure_loaded')
    def test_predict_price_model_loading_error(self, mock_ensure_loaded):
        """Test prediction with model loading error."""
        mock_ensure_loaded.side_effect = FileNotFoundError("Model file not found")

        req = OLXPredictionRequest(
            LB=120.0,
            LT=150.0,
            KM=2,
            KT=3,
            kota_kab="Jakarta Selatan",
            provinsi="Jakarta D.K.I.",
            type_="rumah"
        )

        with pytest.raises(FileNotFoundError):
            predict_price(req)

    def test_predict_price_invalid_input(self):
        """Test prediction with invalid input."""
        req = OLXPredictionRequest(
            LB=-10.0,  # Invalid negative value
            LT=150.0,
            KM=2,
            KT=3,
            kota_kab="Jakarta Selatan",
            provinsi="Jakarta D.K.I.",
            type_="rumah"
        )

        # This should raise validation error from Pydantic
        with pytest.raises(ValueError):
            predict_price(req)