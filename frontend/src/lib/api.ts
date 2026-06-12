import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5001';

console.log('API Configuration:', {
  baseURL: API_BASE_URL,
  envVar: process.env.NEXT_PUBLIC_API_URL
});

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface StockRecommendation {
  symbol: string;
  company_name: string;
  technical_score: number;
  fundamental_score: number;
  sentiment_score: number;
  reason: string;
  backtest_cagr?: number;
  recommendation_date: string;
  combined_score: number;
  recommendation_strength: string;
}

export interface ApiResponse<T> {
  status: 'success' | 'error';
  count?: number;
  recommendations?: T[];
  error?: string;
  message?: string;
}

export interface AnalysisConfig {
  max_stocks?: number;
  test?: boolean;
  all?: boolean;
  group?: string;
  offline?: boolean;
  verbose?: boolean;
  purge_days?: number;
  disable_volume_filter?: boolean;
}

// Get all stock recommendations
export const getRecommendations = async (): Promise<ApiResponse<StockRecommendation>> => {
  try {
    const response = await api.get('/recommendations');
    return response.data;
  } catch (error) {
    console.error('Error fetching recommendations:', error);

    // Return a user-friendly error response instead of throwing
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED' || error.message === 'Network Error') {
        return {
          status: 'error',
          error: 'Backend server not running. Please start the backend server.',
          recommendations: []
        };
      }
      return {
        status: 'error',
        error: error.response?.data?.error || 'Failed to fetch recommendations',
        recommendations: []
      };
    }

    return {
      status: 'error',
      error: 'An unexpected error occurred',
      recommendations: []
    };
  }
};

interface AnalysisResponse {
  message: string;
  config: AnalysisConfig;
}

interface HealthResponse {
  message: string;
  timestamp: string;
}

// Trigger stock analysis
export const triggerAnalysis = async (config: AnalysisConfig = {}): Promise<ApiResponse<AnalysisResponse>> => {
  try {
    const response = await api.post('/trigger-analysis', config);
    return response.data;
  } catch (error) {
    console.error('Error triggering analysis:', error);

    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED' || error.message === 'Network Error') {
        return {
          status: 'error',
          error: 'Backend server not running. Please start the backend server.'
        };
      }
      return {
        status: 'error',
        error: error.response?.data?.error || 'Failed to start analysis'
      };
    }

    return {
      status: 'error',
      error: 'An unexpected error occurred'
    };
  }
};

// Health check
export const healthCheck = async (): Promise<ApiResponse<HealthResponse>> => {
  try {
    const response = await api.get('/');
    return response.data;
  } catch (error) {
    console.error('Error checking health:', error);

    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNREFUSED' || error.message === 'Network Error') {
        return {
          status: 'error',
          error: 'Backend server not running. Please start the backend server.'
        };
      }
      return {
        status: 'error',
        error: error.response?.data?.error || 'Health check failed'
      };
    }

    return {
      status: 'error',
      error: 'An unexpected error occurred'
    };
  }
};

// Get available symbol groups
export const getSymbolGroups = async (): Promise<ApiResponse<string>> => {
  try {
    const response = await api.get('/symbol-groups');
    return {
      status: 'success',
      count: response.data.count,
      recommendations: response.data.groups // Reusing recommendations field for simplicity in ApiResponse
    };
  } catch (error) {
    console.error('Error fetching symbol groups:', error);

    if (axios.isAxiosError(error)) {
      return {
        status: 'error',
        error: error.response?.data?.error || 'Failed to fetch symbol groups',
        recommendations: []
      };
    }

    return {
      status: 'error',
      error: 'An unexpected error occurred',
      recommendations: []
    };
  }
};

// ---------------------------------------------------------------------------
// Trading Cycle (Orchestrator)
// ---------------------------------------------------------------------------

export interface OrchestratorConfig {
  mode?: 'live' | 'replay' | 'date';
  replay_days?: number;
  date?: string;
  verbose?: boolean;
}

export const runOrchestrator = async (config: OrchestratorConfig = {}): Promise<ApiResponse<any>> => {
  try {
    const response = await api.post('/run-orchestrator', config);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return { status: 'error', error: error.response?.data?.error || 'Failed to start trading cycle' };
    }
    return { status: 'error', error: 'An unexpected error occurred' };
  }
};

export const getOrchestratorStatus = async (): Promise<{ status: string; message: string }> => {
  try {
    const response = await api.get('/orchestrator-status');
    return response.data;
  } catch {
    return { status: 'error', message: 'Failed to fetch status' };
  }
};

// ---------------------------------------------------------------------------
// Positions
// ---------------------------------------------------------------------------

export interface Position {
  _id: string;
  symbol: string;
  quantity: number;
  initial_quantity?: number;
  entry_price: number;
  current_price?: number;
  total_investment: number;
  allocation_pct?: number;
  stop_loss: number;
  current_stop_loss?: number;
  target: number;
  current_target?: number;
  strategy_name: string;
  entry_date: string;
  status: string;
  adds_count?: number;
  targets_hit?: number;
  current_target_idx?: number;
  partial_exits?: any[];
  updates?: any[];
  exit_price?: number;
  exit_reason?: string;
  exit_date?: string;
  pnl_pct?: number;
  created_at: string;
  updated_at: string;
  is_paper?: boolean;
  trade_type?: string;
}

export interface CreatePositionData {
  symbol: string;
  quantity: number;
  entry_price: number;
  stop_loss?: number;
  target?: number;
  strategy_name?: string;
  total_investment?: number;
}

export const getPositions = async (status?: string): Promise<{ status: string; count: number; positions: Position[]; error?: string }> => {
  try {
    const params = status ? { status } : {};
    const response = await api.get('/positions', { params });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return { status: 'error', count: 0, positions: [], error: error.response?.data?.error || 'Failed to fetch positions' };
    }
    return { status: 'error', count: 0, positions: [], error: 'Failed to connect to server' };
  }
};

export const createPosition = async (data: CreatePositionData): Promise<ApiResponse<any>> => {
  try {
    const response = await api.post('/positions', data);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return { status: 'error', error: error.response?.data?.error || 'Failed to create position' };
    }
    return { status: 'error', error: 'An unexpected error occurred' };
  }
};

export const updatePosition = async (symbol: string, data: Record<string, any>): Promise<ApiResponse<any>> => {
  try {
    const response = await api.patch(`/positions/${symbol}`, data);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return { status: 'error', error: error.response?.data?.error || 'Failed to update position' };
    }
    return { status: 'error', error: 'An unexpected error occurred' };
  }
};

export const closePosition = async (symbol: string): Promise<ApiResponse<any>> => {
  try {
    const response = await api.delete(`/positions/${symbol}`);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return { status: 'error', error: error.response?.data?.error || 'Failed to close position' };
    }
    return { status: 'error', error: 'An unexpected error occurred' };
  }
};

export const getActivityLogs = async (symbol?: string, limit: number = 100): Promise<{ status: string; count: number; logs: any[]; error?: string }> => {
  try {
    const params: Record<string, any> = { limit };
    if (symbol) params.symbol = symbol;
    const response = await api.get('/activity-logs', { params });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return { status: 'error', count: 0, logs: [], error: error.response?.data?.error || 'Failed to fetch activity logs' };
    }
    return { status: 'error', count: 0, logs: [], error: 'Failed to connect to server' };
  }
};

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

export interface Strategy {
  name: string;
  enabled: boolean;
  description: string;
  version: string;
  file_name: string;
  risk_per_trade_pct?: number;
  max_positions?: number;
  max_position_pct?: number;
  [key: string]: any;
}

export const getStrategies = async (): Promise<{ status: string; count: number; strategies: Strategy[]; error?: string }> => {
  try {
    const response = await api.get('/strategies');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return { status: 'error', count: 0, strategies: [], error: error.response?.data?.error || 'Failed to fetch strategies' };
    }
    return { status: 'error', count: 0, strategies: [], error: 'Failed to connect to server' };
  }
};

// ---------------------------------------------------------------------------
// Trading Config
// ---------------------------------------------------------------------------

export interface TradingConfig {
  is_paper_trading: boolean;
  initial_capital: number;
  brokerage_charges: number;
  auto_execute: boolean;
  circuit_breaker: boolean;
  allow_multiple_positions_same_stock?: boolean;
  time_stop_days?: number;
}

export const getTradingConfig = async (): Promise<{ status: string; config: TradingConfig; error?: string }> => {
  try {
    const response = await api.get('/settings/trading');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return { status: 'error', config: {} as TradingConfig, error: error.response?.data?.error || 'Failed to fetch config' };
    }
    return { status: 'error', config: {} as TradingConfig, error: 'Failed to connect to server' };
  }
};

// ---------------------------------------------------------------------------
// Cycle Stats
// ---------------------------------------------------------------------------

export interface CycleStats {
  open_positions: number;
  total_invested: number;
  cash_remaining: number;
  total_equity: number;
  pnl_pct: number;
  initial_capital: number;
}

export const getCycleStats = async (): Promise<{ status: string; error?: string } & Partial<CycleStats>> => {
  try {
    const response = await api.get('/cycle-stats');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return { status: 'error', error: error.response?.data?.error || 'Failed to fetch stats' };
    }
    return { status: 'error', error: 'Failed to connect to server' };
  }
};

export interface DashboardStats {
  portfolio: {
    total_equity: number;
    total_invested: number;
    cash_remaining: number;
    deployed_pct: number;
    initial_capital: number;
    total_pnl: number;
    pnl_pct: number;
    realized_pnl: number;
    unrealized_pnl: number;
    open_positions: number;
    total_trades: number;
  };
  performance: {
    win_rate: number;
    profit_factor: number;
    avg_win_pct: number;
    avg_loss_pct: number;
    wins: number;
    losses: number;
    total_closed: number;
  };
  today: {
    trades_opened: number;
    pyramids_added: number;
    sl_trails: number;
    targets_hit: number;
    positions_closed: number;
  };
  positions: Array<{
    symbol: string;
    quantity: number;
    entry_price: number;
    current_price: number;
    total_investment: number;
    unrealized_pnl: number;
    pnl_pct: number;
    stop_loss: number;
    target: number;
    adds_count: number;
    strategy: string;
  }>;
  activity_feed: Array<{
    symbol: string;
    action: string;
    timestamp: string;
    details: Record<string, any>;
  }>;
}

export const getDashboardStats = async (): Promise<{ status: string; error?: string } & Partial<DashboardStats>> => {
  try {
    const response = await api.get('/dashboard-stats');
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      return { status: 'error', error: error.response?.data?.error || 'Failed to fetch dashboard stats' };
    }
    return { status: 'error', error: 'Failed to connect to server' };
  }
};
