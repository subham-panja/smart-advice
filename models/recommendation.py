from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any, List

class RecommendedShare(BaseModel):
    """Model for recommended stock shares."""
    id: Optional[int] = None  # Auto-incremented in DB
    symbol: str
    company_name: str
    technical_score: float
    fundamental_score: float
    sentiment_score: float
    recommendation_date: datetime = Field(default_factory=datetime.now)
    reason: str
    buy_price: Optional[float] = None
    sell_price: Optional[float] = None
    est_time_to_target: Optional[str] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_dict(self) -> dict:
        """Convert model to dictionary for database insertion."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'company_name': self.company_name,
            'technical_score': self.technical_score,
            'fundamental_score': self.fundamental_score,
            'sentiment_score': self.sentiment_score,
            'recommendation_date': self.recommendation_date.isoformat(),
            'reason': self.reason,
            'buy_price': self.buy_price,
            'sell_price': self.sell_price,
            'est_time_to_target': self.est_time_to_target
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'RecommendedShare':
        """Create model instance from dictionary (e.g., from database)."""
        if isinstance(data.get('recommendation_date'), str):
            data['recommendation_date'] = datetime.fromisoformat(data['recommendation_date'])
        return cls(**data)


class BacktestResult(BaseModel):
    """Model for backtest results."""
    id: Optional[int] = None  # Auto-incremented in DB
    symbol: str
    period: str
    CAGR: Optional[float] = None
    win_rate: Optional[float] = None
    max_drawdown: Optional[float] = None
    total_trades: Optional[int] = None
    winning_trades: Optional[int] = None
    losing_trades: Optional[int] = None
    avg_trade_duration: Optional[float] = None
    avg_profit_per_trade: Optional[float] = None
    avg_loss_per_trade: Optional[float] = None
    largest_win: Optional[float] = None
    largest_loss: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    volatility: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: Optional[float] = None
    final_capital: Optional[float] = None
    total_return: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def to_dict(self) -> dict:
        """Convert model to dictionary for database insertion."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'period': self.period,
            'CAGR': self.CAGR,
            'win_rate': self.win_rate,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_trade_duration': self.avg_trade_duration,
            'avg_profit_per_trade': self.avg_profit_per_trade,
            'avg_loss_per_trade': self.avg_loss_per_trade,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'volatility': self.volatility,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BacktestResult':
        """Create model instance from dictionary (e.g., from database)."""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class BacktestInfo(BaseModel):
    """Model for comprehensive backtest information."""
    symbol: str
    overall_metrics: Optional[BacktestResult] = None
    period_results: Optional[dict] = None
    summary: Optional[str] = None
    performance_grade: Optional[str] = None
    risk_assessment: Optional[str] = None
    strategy_effectiveness: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        return {
            'symbol': self.symbol,
            'overall_metrics': self.overall_metrics.to_dict() if self.overall_metrics else None,
            'period_results': self.period_results,
            'summary': self.summary,
            'performance_grade': self.performance_grade,
            'risk_assessment': self.risk_assessment,
            'strategy_effectiveness': self.strategy_effectiveness
        }
