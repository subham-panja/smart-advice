from pydantic import BaseModel, Field
from datetime import date
from typing import List, Optional

class StockData(BaseModel):
    """Model for individual stock OHLCV data point."""
    open: float
    high: float
    low: float
    close: float
    volume: int
    date: date  # Date of the OHLCV data

class StockDetails(BaseModel):
    """Model for complete stock information."""
    symbol: str
    company_name: str
    industry: Optional[str] = None
    sector: Optional[str] = None
    historical_data: List[StockData] = Field(default_factory=list)
