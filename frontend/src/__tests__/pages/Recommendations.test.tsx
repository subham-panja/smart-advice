import { render, screen, waitFor } from '@testing-library/react';
import RecommendationsPage from '../../app/recommendations/page';
import * as api from '../../lib/api';

jest.mock('../../lib/api', () => ({
  getRecommendations: jest.fn(),
}));

const mockedApi = api as jest.Mocked<typeof api>;

const mockRecommendations = [
  {
    symbol: 'WABAG',
    company_name: 'VA Tech Wabag',
    technical_score: 7.5,
    fundamental_score: 6.0,
    sentiment_score: 5.0,
    combined_score: 6.5,
    recommendation_strength: 'BUY',
    recommendation_date: '2026-01-01',
    backtest_cagr: 15.2,
  },
];

beforeEach(() => {
  (mockedApi.getRecommendations as jest.Mock).mockResolvedValue({
    status: 'success',
    count: 1,
    recommendations: mockRecommendations,
  });
});

describe('Recommendations Page', () => {
  it('renders page title', () => {
    render(<RecommendationsPage />);
    expect(screen.getByText('Stock Recommendations')).toBeInTheDocument();
  });

  it('renders refresh button', () => {
    render(<RecommendationsPage />);
    expect(screen.getByText('Refresh')).toBeInTheDocument();
  });

  it('loads and displays recommendations', async () => {
    render(<RecommendationsPage />);
    await waitFor(() => {
      expect(screen.getByText('WABAG')).toBeInTheDocument();
    });
  });

  it('shows error state on failure', async () => {
    (mockedApi.getRecommendations as jest.Mock).mockResolvedValue({
      status: 'error',
      error: 'Server down',
    });
    render(<RecommendationsPage />);
    await waitFor(() => {
      expect(screen.getByText('Server down')).toBeInTheDocument();
    });
  });

  it('shows empty state when no recommendations', async () => {
    (mockedApi.getRecommendations as jest.Mock).mockResolvedValue({
      status: 'success',
      count: 0,
      recommendations: [],
    });
    render(<RecommendationsPage />);
    await waitFor(() => {
      expect(screen.getByText('No recommendations available.')).toBeInTheDocument();
    });
  });
});
