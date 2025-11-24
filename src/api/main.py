# fastapi_app/main.py
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# File: app.py (PASTIKAN NAMA FILENYA TETAP app.py)

from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World, API Houseindo is running!"}

# Opsional: Biar bisa dijalankan pakai python app.py juga
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- Fallback utk pipeline lama yang expect fastapi_app.main._make_interactions
def _make_interactions(df):
    """
    Implementasi minimal agar unpickle FunctionTransformer tidak error.
    Ganti sesuai logika training aslinya bila ada.
    """
    try:
        df = df.copy()
        if "LB" in df.columns and "LT" in df.columns:
            df["LBxLT"] = df["LB"] * df["LT"]     # contoh interaksi umum
            # jika dulu ada rasio: df["LB_per_LT"] = df["LB"] / df["LT"].replace(0, pd.NA)
    except Exception:
        pass
    return df
# ---- end fallback

from .schemas import OLXPredictionRequest, PredictionResponse
from .inference import predict_price

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
    description="API for predicting house prices using machine learning models"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for better error responses."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

@app.get("/health")
def health():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "ok", "service": "house-price-prediction-api"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: OLXPredictionRequest):
    """
    Predict house price based on input features.

    This endpoint accepts house property details and returns a price prediction
    along with confidence metrics and feature importance.
    """
    try:
        logger.info(f"Prediction request received: {req.dict()}")
        result = predict_price(req)
        logger.info(f"Prediction completed successfully: Rp {result.prediction:,.0f}")
        return result
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
