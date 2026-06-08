import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import PositionsPage from '../../app/positions/page';
import * as api from '../../lib/api';

jest.mock('../../lib/api', () => ({
  getPositions: jest.fn(),
  updatePosition: jest.fn(),
  closePosition: jest.fn(),
}));

const mockedApi = api as jest.Mocked<typeof api>;

const mockPositions = [
  {
    _id: '1',
    symbol: 'WABAG',
    quantity: 6,
    entry_price: 1500.0,
    current_stop_loss: 1450.0,
    current_target: 1650.0,
    total_investment: 9000,
    strategy_name: 'Swing_Trading',
    stop_loss: 1450.0,
    target: 1650.0,
    status: 'OPEN',
    created_at: '2026-01-01T00:00:00',
    updated_at: '2026-01-01T00:00:00',
  },
];

beforeEach(() => {
  (mockedApi.getPositions as jest.Mock).mockResolvedValue({
    status: 'success',
    count: 1,
    positions: mockPositions,
  });
  (mockedApi.updatePosition as jest.Mock).mockResolvedValue({ status: 'success', message: 'Updated' });
  (mockedApi.closePosition as jest.Mock).mockResolvedValue({ status: 'success', message: 'Closed' });
});

afterEach(() => jest.clearAllMocks());

describe('Positions Page', () => {
  it('renders the page title', async () => {
    render(<PositionsPage />);
    expect(screen.getByText('Positions')).toBeInTheDocument();
  });

  it('shows refresh button', () => {
    render(<PositionsPage />);
    expect(screen.getByText('Refresh')).toBeInTheDocument();
  });

  it('renders positions table with data', async () => {
    render(<PositionsPage />);
    await waitFor(() => {
      expect(screen.getByText('WABAG')).toBeInTheDocument();
    });
  });

  it('shows position count in header', async () => {
    render(<PositionsPage />);
    await waitFor(() => {
      expect(screen.getByText('Open Positions (1)')).toBeInTheDocument();
    });
  });

  it('shows empty state when no positions', async () => {
    (mockedApi.getPositions as jest.Mock).mockResolvedValue({ status: 'success', count: 0, positions: [] });
    render(<PositionsPage />);
    await waitFor(() => {
      expect(screen.getByText('No open positions.')).toBeInTheDocument();
    });
  });

  it('opens edit modal when edit button clicked', async () => {
    render(<PositionsPage />);
    await waitFor(() => expect(screen.getByText('WABAG')).toBeInTheDocument());

    const editBtns = screen.getAllByTitle('Edit');
    fireEvent.click(editBtns[0]);

    expect(screen.getByText('Edit Position')).toBeInTheDocument();
    expect(screen.getAllByText('WABAG').length).toBeGreaterThanOrEqual(1);
  });

  it('does not have Add Position button', () => {
    render(<PositionsPage />);
    expect(screen.queryByText('Add Position')).not.toBeInTheDocument();
  });
});
