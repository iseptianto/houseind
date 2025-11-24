from pydantic import BaseModel, Field
from typing import Optional


class OLXPredictionRequest(BaseModel):
    """
    Request model for house price prediction.
    
    All measurements should be in standard units (meters, square meters).
    """
    LB: float = Field(
        ...,
        gt=0,
        description="Luas Bangunan (Building Area) in square meters",
        examples=[120.0, 200.0, 150.0]
    )
    LT: float = Field(
        ...,
        gt=0,
        description="Luas Tanah (Land Area) in square meters",
        examples=[150.0, 300.0, 200.0]
    )
    KM: int = Field(
        ..., 
        ge=0,
        description="Kamar Mandi (Number of bathrooms)"
    )
    KT: int = Field(
        ..., 
        ge=0,
        description="Kamar Tidur (Number of bedrooms)"
    )

    # Location fields with proper aliases
    kota_kab: str = Field(
        ...,
        alias="Kota/Kab",
        description="City/Regency name",
        examples=["Jakarta Selatan", "Bandung", "Surabaya"]
    )
    provinsi: str = Field(
        ...,
        alias="Provinsi",
        description="Province name",
        examples=["Jakarta D.K.I.", "Jawa Barat", "Jawa Timur"]
    )
    type_: str = Field(
        ...,
        alias="Type",
        description="Property type/category",
        examples=["rumah", "apartemen"]
    )

    # Optional fields
    harga_per_m2: Optional[float] = Field(
        None, 
        alias="harga_per_m2",
        ge=0,
        description="Price per square meter (if known)"
    )
    ratio_bangunan_rumah: Optional[float] = Field(
        None, 
        alias="ratio_bangunan_rumah",
        ge=0,
        le=1,
        description="Building to land ratio (if known)"
    )

    class Config:
        allow_population_by_alias = True
        anystr_strip_whitespace = True


class PredictionResponse(BaseModel):
    """Response model for house price prediction."""
    prediction: float = Field(
        ..., 
        description="Predicted house price in the same currency as training data"
    )
    prediction_time: str = Field(
        ..., 
        description="UTC timestamp of when the prediction was made"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the prediction (0-1)"
    )
    model_name: str = Field(
        ...,
        description="Name of the model used for prediction"
    )
    price_range: tuple[float, float] = Field(
        ...,
        description="Estimated price range (min, max)"
    )
    feature_importance: dict[str, float] = Field(
        ...,
        description="Top features affecting the price prediction"
    )
    prediction_time_ms: float = Field(
        ...,
        description="Time taken to make prediction in milliseconds"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1250000000.0,
                "prediction_time": "2025-10-22T01:23:45.678Z",
                "confidence_score": 0.92,
                "model_name": "XGBoost",
                "price_range": [1150000000.0, 1350000000.0],
                "feature_importance": {
                    "Square Footage": 0.35,
                    "Location": 0.27,
                    "Number of Bathrooms": 0.15
                },
                "prediction_time_ms": 120.5
            }
        }
