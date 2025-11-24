# fastapi_app/inference.py
import os
import joblib
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import logging

from .schemas import OLXPredictionRequest, PredictionResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
# Default paths for models in container
DEFAULT_MODEL_PATH = Path("models/trained/modelbaru.pkl")
DEFAULT_PREP_PATH = Path("models/trained/barupreprocessor.pkl")

# Allow overriding via env vars
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", str(DEFAULT_PREP_PATH)))

# For local development, try relative paths if absolute paths don't exist
if not MODEL_PATH.exists() and not str(MODEL_PATH).startswith('/'):
    local_model_path = BASE_DIR.parent.parent / "models" / "modelbaru.pkl"
    if local_model_path.exists():
        MODEL_PATH = local_model_path
        logger.info(f"Using local model path: {MODEL_PATH}")

if not PREPROCESSOR_PATH.exists() and not str(PREPROCESSOR_PATH).startswith('/'):
    local_prep_path = BASE_DIR.parent.parent / "models" / "barupreprocessor.pkl"
    if local_prep_path.exists():
        PREPROCESSOR_PATH = local_prep_path
        logger.info(f"Using local preprocessor path: {PREPROCESSOR_PATH}")

# Validate environment variables
def validate_env_vars():
    """Validate required environment variables for inference."""
    if not MODEL_PATH.exists():
        logger.warning(f"Model path {MODEL_PATH} does not exist. This may cause issues during deployment.")
    if not PREPROCESSOR_PATH.exists():
        logger.warning(f"Preprocessor path {PREPROCESSOR_PATH} does not exist. This may cause issues during deployment.")

    return MODEL_PATH, PREPROCESSOR_PATH

MODEL_PATH, PREPROCESSOR_PATH = validate_env_vars()

_model = None
_preproc = None

def _ensure_loaded():
    global _model, _preproc
    if _model is None or _preproc is None:
        logger.info("Loading model and preprocessor...")

        # Some preprocessor objects may have been pickled when a helper
        # function (_make_interactions) was defined in a training script
        # run as __main__. During unpickling we must ensure that symbol
        # exists on the same module. Provide a safe fallback here.
        def _make_interactions(df):
            try:
                df = df.copy()
                if "LB" in df.columns and "LT" in df.columns:
                    df["LBxLT"] = df["LB"] * df["LT"]
            except Exception:
                pass
            return df

        # Inject into __main__ so pickle can find it if it was saved from a
        # script executed as __main__ previously.
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and not hasattr(main_mod, "_make_interactions"):
            setattr(main_mod, "_make_interactions", _make_interactions)

        # Provide helpful errors when model files are missing
        if not PREPROCESSOR_PATH.exists():
            error_msg = f"Preprocessor file not found: {PREPROCESSOR_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not MODEL_PATH.exists():
            error_msg = f"Model file not found: {MODEL_PATH}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading preprocessor from {PREPROCESSOR_PATH}")
            _preproc = joblib.load(PREPROCESSOR_PATH)
            logger.info("Preprocessor loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load preprocessor from {PREPROCESSOR_PATH}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            _model = joblib.load(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load model from {MODEL_PATH}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

CSV_COLS = [
    "LB","LT","KM","KT","Kota/Kab","Provinsi","Type"
]

def _engineer_features(df):
    """Create all required features for the model."""
    try:
        # Basic features
        df['LBxLT'] = df['LB'] * df['LT']
        df['log_LB'] = np.log1p(df['LB'])
        df['log_LT'] = np.log1p(df['LT'])
        df['lb_x_km'] = df['LB'] * df['KM']
        df['lt_x_kt'] = df['LT'] * df['KT']
        # Handle division by zero for ratio
        df['ratio_lb_lt'] = df['LB'] / df['LT'].replace(0, np.nan)

        # Keep original features needed by preprocessor
        required_cols = [
            'LB', 'LT', 'KM', 'KT', 'Kota/Kab', 'Provinsi', 'Type',
            'LBxLT', 'log_LB', 'log_LT', 'lb_x_km', 'lt_x_kt', 'ratio_lb_lt'
        ]

        return df[required_cols]
    except Exception as e:
        raise ValueError(f"Error engineering features: {str(e)}")

def _to_row(req: OLXPredictionRequest) -> dict:
    """Convert request to initial dataframe row."""
    try:
        r = req.dict(by_alias=True)
        base = {c: r.get(c, None) for c in CSV_COLS}
        return base
    except Exception as e:
        raise ValueError(f"Error converting request to row: {str(e)}")

def predict_price(req: OLXPredictionRequest) -> PredictionResponse:
    """
    Generate house price prediction from input features.

    Args:
        req: Validated request containing house features

    Returns:
        PredictionResponse with prediction details and confidence metrics

    Raises:
        ValueError: If feature engineering fails
        RuntimeError: If model prediction fails
    """
    try:
        logger.info(f"Processing prediction request for property in {req.kota_kab}, {req.provinsi}")
        start_time = datetime.now()
        _ensure_loaded()

        # Create initial dataframe
        row_dict = _to_row(req)
        df = pd.DataFrame([row_dict])

        # Engineer features
        df = _engineer_features(df)
        logger.debug(f"Engineered features: {list(df.columns)}")

        # Generate prediction
        try:
            X = _preproc.transform(df)
            logger.debug(f"Transformed features shape: {X.shape}")

            # Get base prediction
            y = _model.predict(X)
            price = float(y[0])

            # Ensure prediction is non-negative
            price = max(0, price)
            logger.info(f"Predicted price: Rp {price:,.0f}")

            # Get confidence score (using predict_proba if available, else use a heuristic)
            confidence_score = 0.85  # Default value for regression models
            if hasattr(_model, 'predict_proba'):
                try:
                    proba = _model.predict_proba(X)
                    confidence_score = float(proba.max())
                    logger.debug(f"Confidence score from predict_proba: {confidence_score}")
                except Exception as e:
                    logger.warning(f"Could not get confidence from predict_proba: {e}")
                    confidence_score = 0.85

            # Calculate price range (±10% by default)
            price_range = (price * 0.9, price * 1.1)

            # Get feature importance
            feature_importance = {}
            if hasattr(_model, 'feature_importances_'):
                importances = _model.feature_importances_
                # Get feature names from preprocessor if available
                try:
                    if hasattr(_preproc, 'get_feature_names_out'):
                        feature_names = _preproc.get_feature_names_out()
                        logger.debug(f"Feature names from preprocessor: {feature_names[:5]}...")
                    else:
                        # Fallback to generic names
                        feature_names = [f"Feature_{i}" for i in range(len(importances))]
                except Exception as e:
                    logger.warning(f"Could not get feature names: {e}")
                    # Fallback to predefined names
                    feature_names = [
                        "Square Footage (LB)",
                        "Location",
                        "Number of Bathrooms",
                        "Land Area (LT)",
                        "Number of Bedrooms"
                    ]

                importance_dict = dict(zip(feature_names, importances))
                # Sort by importance and get top 3
                feature_importance = dict(sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3])
                logger.debug(f"Top 3 important features: {list(feature_importance.keys())}")

            # Calculate prediction time
            end_time = datetime.now()
            prediction_time_ms = (end_time - start_time).total_seconds() * 1000
            logger.info(f"Prediction completed in {prediction_time_ms:.2f}ms")

            # Get model name
            model_name = type(_model).__name__
            if model_name == 'XGBRegressor':
                model_name = 'XGBoost'

            return PredictionResponse(
                prediction=round(price, 2),
                prediction_time=datetime.utcnow().isoformat() + "Z",
                confidence_score=confidence_score,
                model_name=model_name,
                price_range=price_range,
                feature_importance=feature_importance,
                prediction_time_ms=prediction_time_ms
            )
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Error during prediction: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise ValueError(f"Error processing request: {str(e)}")
    # This code block appears to be duplicate/unused code that was accidentally left in the file
    # It should be removed as it's redundant with the main predict_price function above
