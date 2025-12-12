from typing import List, Literal

from pydantic import BaseModel, Field


class CryptoAnalysisRequest(BaseModel):
    coins: List[str] = Field(..., description="Provide the list of coins to be analyzed")


class CryptoCompareRequest(CryptoAnalysisRequest):
    pass


class CryptoComparison(BaseModel):
    winner: str
    summary: str
    reasons: List[str]


class CryptoComparisonResponse(BaseModel):
    comparison: CryptoComparison


class MarketFactor(BaseModel):
    factor: str
    impact: str


class CryptoInsights(BaseModel):
    prediction: str
    confidence: int = Field(..., le=100, ge=0)


class CoinMarketAnalysis(BaseModel):
    coin: str
    summary: str
    sentiment: Literal["bullish", "bearish", "neutral"]
    key_factors: List[MarketFactor]
    insights: List[CryptoInsights]


class CryptoAnalysisResponse(BaseModel):
    analysis: List[CoinMarketAnalysis]


class CryptoRequest(BaseModel):
    coin_list: List[str]
