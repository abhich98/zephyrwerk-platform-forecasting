from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class PriceRequest(BaseModel):
    target_date: date
    hour: Optional[int] = Field(default=None, ge=0, le=23, description="Hour of day (0-23)")

class GenerationRequest(BaseModel):
    target_date: date
    hour: Optional[int] = Field(default=None, ge=0, le=23, description="Hour of day (0-23)")
