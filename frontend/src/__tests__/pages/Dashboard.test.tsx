import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import Home from '../../app/page';
import * as api from '../../lib/api';

// Mock dependencies
jest.mock('../../lib/api', () => ({
  getDashboardStats: jest.fn(),
  getRecommendations: jest.fn(),
  getDailyEarnings: jest.fn(),
  getStockCandles: jest.fn(),
  updateRecommendation: jest.fn(),
  updatePosition: jest.fn(),
  getCycleStats: jest.fn(),
}));

jest.mock('react-chartjs-2', () => ({
  Bar: () => <div data-testid="bar-chart">Bar Chart</div>,
  Doughnut: () => <div data-testid="doughnut-chart">Doughnut Chart</div>,
}));

// Mock ResizeObserver for jsdom
beforeAll(() => {
  global.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

const mockRecommendations = [
  {
    symbol: 'WABAG',
    company_name: 'VA Tech Wabag Ltd',
    technical_score: 8.2,
    fundamental_score: 7.5,
    sentiment_score: 6.8,
    combined_score: 0.85,
    composite_score: 8.5,
    buy_price: 1250,
    sell_price: 1500,
    stop_loss: 1180,
    rr_ratio: 3.5,
    suggested_quantity: 40,
    allocation_pct: 5.0,
    entry_atr: 35.5,
    recommendation_strength: 'BUY',
    recommendation_date: '2026-03-01',
    reason: 'Minervini VCP breakout with expanding institutional volume flow',
    pattern_detected: 'VCP Breakout',
    note: 'Initial breakout pivot test',
  },
  {
    symbol: 'TCS',
    company_name: 'Tata Consultancy Services',
    technical_score: 7.9,
    fundamental_score: 8.5,
    sentiment_score: 7.2,
    combined_score: 0.82,
    composite_score: 8.2,
    buy_price: 3800,
    sell_price: 4200,
    stop_loss: 3650,
    rr_ratio: 2.7,
    suggested_quantity: 15,
    allocation_pct: 5.7,
    entry_atr: 70.0,
    recommendation_strength: 'BUY',
    recommendation_date: '2026-03-01',
    reason: 'Orderly pullback to 21 EMA support on low volume',
    pattern_detected: '21 EMA Pullback',
  },
];

const mockStats = {
  status: 'success',
  portfolio: {
    total_equity: 125000,
    total_invested: 85000,
    cash_remaining: 40000,
    deployed_pct: 68.0,
    initial_capital: 100000,
    total_pnl: 25000,
    pnl_pct: 25.0,
    realized_pnl: 15000,
    unrealized_pnl: 10000,
    open_positions: 2,
    total_trades: 12,
  },
  performance: {
    win_rate: 68.5,
    profit_factor: 2.45,
    avg_win_pct: 8.2,
    avg_loss_pct: 2.8,
    wins: 8,
    losses: 4,
    total_closed: 12,
  },
  today: {
    trades_opened: 1,
    pyramids_added: 0,
    sl_trails: 2,
    targets_hit: 1,
    positions_closed: 0,
  },
  positions: [
    {
      symbol: 'WABAG',
      quantity: 40,
      entry_price: 1200,
      current_price: 1350,
      total_investment: 48000,
      unrealized_pnl: 6000,
      pnl_pct: 12.5,
      adds_count: 1,
      strategy: 'VCP_Breakout',
      reason: 'Stage 2 base breakout',
      note: 'Added +1 on pullback confirmation',
      stop_loss: 1180,
      target: 1500,
      entry_atr: 35.0,
    },
  ],
  activity_feed: [
    {
      symbol: 'WABAG',
      action: 'TRAIL_SL',
      timestamp: new Date().toISOString(),
    },
  ],
};

beforeEach(() => {
  jest.clearAllMocks();
  (api.getDashboardStats as jest.Mock).mockResolvedValue(mockStats);
  (api.getRecommendations as jest.Mock).mockResolvedValue({
    status: 'success',
    recommendations: mockRecommendations,
  });
  (api.getDailyEarnings as jest.Mock).mockResolvedValue({
    status: 'success',
    earnings: [],
    summary: {},
  });
  (api.getStockCandles as jest.Mock).mockResolvedValue({
    status: 'success',
    candles: [],
  });
  (api.updateRecommendation as jest.Mock).mockResolvedValue({ status: 'success' });
  (api.updatePosition as jest.Mock).mockResolvedValue({ status: 'success' });
});

describe('Dashboard Page', () => {
  it('renders the Dashboard header and subtitle', async () => {
    render(<Home />);
    await waitFor(() => {
      expect(screen.getByRole('heading', { level: 1, name: 'Dashboard' })).toBeInTheDocument();
      expect(screen.getByText(/Portfolio overview/i)).toBeInTheDocument();
    });
  });

  it('renders the KPI cards with formatted values', async () => {
    render(<Home />);
    await waitFor(() => {
      expect(screen.getByText('Total Equity')).toBeInTheDocument();
      expect(screen.getByText('₹1.25L')).toBeInTheDocument();
      expect(screen.getByText('Total P&L')).toBeInTheDocument();
      expect(screen.getByText('Cash Available')).toBeInTheDocument();
      expect(screen.getAllByText('Open Positions').length).toBeGreaterThan(0);
    });
  });

  it('renders the Featured Stock Recommendation & Technical Analysis Deep-Dive section', async () => {
    render(<Home />);
    await waitFor(() => {
      expect(
        screen.getByText(/Featured Stock Recommendation & Technical Analysis Deep-Dive/i)
      ).toBeInTheDocument();
      // Should show the horizontal selector pills with the mocked recommendations
      expect(screen.getByRole('button', { name: /WABAG/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /TCS/i })).toBeInTheDocument();
    });
  });

  it('defaults selectedRecommendation to the first stock and allows toggling to another', async () => {
    render(<Home />);
    await waitFor(() => {
      // Default selected is WABAG (the first recommendation)
      const vcpElements = screen.getAllByText('Minervini VCP Breakout');
      expect(vcpElements.length).toBeGreaterThan(0);
    });

    // Click TCS selector pill
    const tcsButton = screen.getByRole('button', { name: /TCS/i });
    fireEvent.click(tcsButton);

    await waitFor(() => {
      // Should now show TCS analysis
      const emaElements = screen.getAllByText('21 EMA Pullback Reversal');
      expect(emaElements.length).toBeGreaterThan(0);
    });
  });

  it('renders both desktop and mobile views for Open Positions', async () => {
    render(<Home />);
    await waitFor(() => {
      // Both desktop table and mobile card contain WABAG position
      const wabagElements = screen.getAllByText('WABAG');
      expect(wabagElements.length).toBeGreaterThan(0);
      expect(screen.getByText(/1 Active/i)).toBeInTheDocument();
    });
  });

  it('renders Inspect action in Latest Stock Recommendations', async () => {
    render(<Home />);
    await waitFor(() => {
      const inspectButtons = screen.getAllByRole('button', { name: /Inspect/i });
      expect(inspectButtons.length).toBeGreaterThan(0);
    });
  });

  it('renders clean fallback state when no recommendations are returned', async () => {
    (api.getRecommendations as jest.Mock).mockResolvedValue({
      status: 'success',
      recommendations: [],
    });

    render(<Home />);
    await waitFor(() => {
      expect(
        screen.getByText(/No Active Recommendations In Spotlight/i)
      ).toBeInTheDocument();
    });
  });
});
