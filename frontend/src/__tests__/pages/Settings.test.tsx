import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Settings from '../../app/settings/page';
import * as api from '../../lib/api';

jest.mock('../../lib/api', () => ({
  getTradingConfig: jest.fn(),
  getStrategies: jest.fn(),
}));

jest.mock('../../app/contexts/ThemeContext', () => ({
  useTheme: () => ({
    theme: 'light',
    setTheme: jest.fn(),
    toggleTheme: jest.fn(),
    mounted: true,
  }),
}));

const mockedApi = api as jest.Mocked<typeof api>;

beforeEach(() => {
  (mockedApi.getTradingConfig as jest.Mock).mockResolvedValue({
    status: 'success',
    config: {
      is_paper_trading: true,
      initial_capital: 10000,
      brokerage_charges: 0.002,
      auto_execute: true,
      circuit_breaker: false,
      time_stop_days: 15,
      allow_multiple_positions_same_stock: false,
    },
  });
  (mockedApi.getStrategies as jest.Mock).mockResolvedValue({
    status: 'success',
    count: 2,
    strategies: [
      {
        name: 'Swing_Trading',
        enabled: true,
        description: 'Swing trading strategy',
        version: '2.0',
        file_name: 'swing_trading.json',
        risk_management: { risk_per_trade_pct: 2, max_positions: 4, max_position_pct: 30 },
      },
      {
        name: 'Momentum',
        enabled: false,
        description: 'Momentum strategy',
        version: '1.0',
        file_name: 'momentum.json',
        risk_management: { risk_per_trade_pct: 1.5, max_positions: 8, max_position_pct: 20 },
      },
    ],
  });
});

afterEach(() => jest.clearAllMocks());

describe('Settings Page', () => {
  it('renders page title', () => {
    render(<Settings />);
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('renders 3 sections in sidebar', () => {
    render(<Settings />);
    expect(screen.getAllByText('General').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Trading Config')).toBeInTheDocument();
    expect(screen.getByText('Strategies')).toBeInTheDocument();
  });

  it('shows theme buttons in General section', () => {
    render(<Settings />);
    expect(screen.getByText('light')).toBeInTheDocument();
    expect(screen.getByText('dark')).toBeInTheDocument();
    expect(screen.getByText('system')).toBeInTheDocument();
  });

  it('switches to Trading Config section', async () => {
    render(<Settings />);
    fireEvent.click(screen.getByText('Trading Config'));

    await waitFor(() => {
      expect(screen.getByText('Paper Trading')).toBeInTheDocument();
    });
  });

  it('shows read-only notice in Trading Config', async () => {
    render(<Settings />);
    fireEvent.click(screen.getByText('Trading Config'));

    await waitFor(() => {
      expect(screen.getByText(/read-only/)).toBeInTheDocument();
    });
  });

  it('switches to Strategies section', async () => {
    render(<Settings />);
    fireEvent.click(screen.getByText('Strategies'));

    await waitFor(() => {
      expect(screen.getByText('Swing_Trading')).toBeInTheDocument();
    });
  });

  it('opens strategy modal on row click', async () => {
    render(<Settings />);
    fireEvent.click(screen.getByText('Strategies'));

    await waitFor(() => {
      expect(screen.getByText('Swing_Trading')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Swing_Trading'));

    await waitFor(() => {
      expect(screen.getByText('swing_trading.json')).toBeInTheDocument();
    });
  });
});
