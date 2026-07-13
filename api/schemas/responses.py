from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class HourlyPrediction(BaseModel):
    hour: int
    value: float

class PriceResponse(BaseModel):
    target_date: date
    hour: Optional[int] = Field(default=None, ge=0, le=23, description="Hour of day (0-23)")
    prices: list[HourlyPrediction]
    model_trained_at: str | None

class GenerationResponse(BaseModel):
    target_date: date
    hour: Optional[int] = Field(default=None, ge=0, le=23, description="Hour of day (0-23)")
    solar_generations: list[HourlyPrediction]
    wind_generations: list[HourlyPrediction]

class EnergySummary(BaseModel):
    target_date: date
    generation_mix: dict[str, float]
    avg_price: float
    renewable_share: float

class Health(BaseModel):
    """Schema for health check response."""
    status: str
    timestamp: datetime
    database: str

