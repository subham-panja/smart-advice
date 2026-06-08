import { render, screen } from '@testing-library/react';
import Home from '../../app/page';

jest.mock('../../lib/api', () => ({
  getRecommendations: jest.fn().mockResolvedValue({ status: 'success', recommendations: [] }),
  getCycleStats: jest.fn().mockResolvedValue({
    status: 'success',
    open_positions: 3,
    total_equity: 9500,
    cash_remaining: 5000,
    pnl_pct: -5.0,
    initial_capital: 10000,
  }),
}));

jest.mock('react-chartjs-2', () => ({
  Bar: () => <div data-testid="bar-chart">Bar Chart</div>,
  Doughnut: () => <div data-testid="doughnut-chart">Doughnut Chart</div>,
}));

describe('Dashboard Page', () => {
  it('renders the hero title', async () => {
    render(<Home />);
    expect(screen.getByText('Stock Advice Dashboard')).toBeInTheDocument();
  });

  it('renders the subtitle', async () => {
    render(<Home />);
    expect(screen.getByText(/AI-powered stock analysis/)).toBeInTheDocument();
  });

  it('does not have Run Trading Cycle button', () => {
    render(<Home />);
    expect(screen.queryByText('Run Trading Cycle')).not.toBeInTheDocument();
  });
});
