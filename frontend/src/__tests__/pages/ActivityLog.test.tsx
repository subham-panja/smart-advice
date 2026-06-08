import { render, screen, waitFor } from '@testing-library/react';
import ActivityLogPage from '../../app/activity-log/page';
import * as api from '../../lib/api';

jest.mock('../../lib/api', () => ({
  getPositions: jest.fn(),
}));

const mockedApi = api as jest.Mocked<typeof api>;

const mockPositions = [
  {
    _id: '1',
    symbol: 'WABAG',
    quantity: 6,
    entry_price: 1500.0,
    stop_loss: 1450.0,
    target: 1650.0,
    total_investment: 9000,
    strategy_name: 'Swing_Trading',
    status: 'OPEN',
    created_at: '2026-01-01T00:00:00',
    updated_at: '2026-01-01T00:00:00',
    updates: [
      { type: 'TRAIL_SL', date: '2026-01-02T00:00:00', prev_sl: 1450, current_sl: 1475 },
    ],
  },
  {
    _id: '2',
    symbol: 'TATAPOWER',
    quantity: 10,
    entry_price: 400.0,
    stop_loss: 380.0,
    target: 450.0,
    total_investment: 4000,
    strategy_name: 'Swing_Trading',
    status: 'CLOSED',
    created_at: '2026-01-01T00:00:00',
    updated_at: '2026-01-05T00:00:00',
    exit_date: '2026-01-05T00:00:00',
    exit_price: 420.0,
    exit_reason: 'MANUAL_CLOSE',
    pnl_pct: 5.0,
    updates: [],
  },
];

beforeEach(() => {
  (mockedApi.getPositions as jest.Mock).mockResolvedValue({
    status: 'success',
    count: 2,
    positions: mockPositions,
  });
});

describe('Activity Log Page', () => {
  it('renders page title', () => {
    render(<ActivityLogPage />);
    expect(screen.getByText('Position Activity Log')).toBeInTheDocument();
  });

  it('renders refresh button', () => {
    render(<ActivityLogPage />);
    expect(screen.getByText('Refresh')).toBeInTheDocument();
  });

  it('shows selector dropdown', async () => {
    render(<ActivityLogPage />);
    await waitFor(() => {
      const select = screen.getByRole('combobox');
      expect(select).toBeInTheDocument();
    });
  });

  it('shows empty state when no position selected', async () => {
    render(<ActivityLogPage />);
    await waitFor(() => {
      expect(screen.getByText(/Select a position from the dropdown/)).toBeInTheDocument();
    });
  });

  it('populates dropdown with positions', async () => {
    render(<ActivityLogPage />);
    await waitFor(() => {
      const select = screen.getByRole('combobox');
      const options = select.querySelectorAll('option');
      expect(options.length).toBe(3); // "Choose" + 2 positions
    });
  });

  it('shows empty state when no positions exist', async () => {
    (mockedApi.getPositions as jest.Mock).mockResolvedValue({ status: 'success', count: 0, positions: [] });
    render(<ActivityLogPage />);
    await waitFor(() => {
      expect(screen.getByText(/No positions found/)).toBeInTheDocument();
    });
  });
});
