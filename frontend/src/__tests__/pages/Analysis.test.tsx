import { render, screen, fireEvent } from '@testing-library/react';
import TradingCyclePage from '../../app/analysis/page';
import * as api from '../../lib/api';

jest.mock('../../lib/api', () => ({
  runOrchestrator: jest.fn(),
  getOrchestratorStatus: jest.fn(),
}));

const mockedApi = api as jest.Mocked<typeof api>;

beforeEach(() => {
  (mockedApi.runOrchestrator as jest.Mock).mockResolvedValue({ status: 'success', message: 'Started' });
  (mockedApi.getOrchestratorStatus as jest.Mock).mockResolvedValue({ status: 'idle', message: '' });
});

afterEach(() => jest.clearAllMocks());

describe('Trading Cycle Page', () => {
  it('renders hero section', () => {
    render(<TradingCyclePage />);
    expect(screen.getByRole('heading', { name: 'Run Trading Cycle' })).toBeInTheDocument();
  });

  it('renders 3 mode cards', () => {
    render(<TradingCyclePage />);
    expect(screen.getByText('Paper / Live')).toBeInTheDocument();
    expect(screen.getByText('Replay')).toBeInTheDocument();
    expect(screen.getByText('Specific Date')).toBeInTheDocument();
  });

  it('renders verbose checkbox', () => {
    render(<TradingCyclePage />);
    expect(screen.getByText('Verbose Logging')).toBeInTheDocument();
  });

  it('renders run button', () => {
    render(<TradingCyclePage />);
    const buttons = screen.getAllByText('Run Trading Cycle');
    expect(buttons.length).toBeGreaterThanOrEqual(1);
  });

  it('shows replay days input when replay mode selected', () => {
    render(<TradingCyclePage />);
    fireEvent.click(screen.getByText('Replay'));
    expect(screen.getByPlaceholderText('e.g. 5')).toBeInTheDocument();
  });

  it('shows date input when date mode selected', () => {
    render(<TradingCyclePage />);
    fireEvent.click(screen.getByText('Specific Date'));
    expect(screen.getByDisplayValue('')).toBeInTheDocument();
  });

  it('calls runOrchestrator when run button clicked', async () => {
    global.EventSource = jest.fn(() => ({
      onmessage: null,
      onerror: null,
      close: jest.fn(),
      readyState: 0,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
    })) as any;

    render(<TradingCyclePage />);
    const runButtons = screen.getAllByText('Run Trading Cycle');
    const btn = runButtons.find(el => el.tagName === 'SPAN')?.parentElement;
    if (btn) fireEvent.click(btn);

    expect(mockedApi.runOrchestrator).toHaveBeenCalled();
  });
});
