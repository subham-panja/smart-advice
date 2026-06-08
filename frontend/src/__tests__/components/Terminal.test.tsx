import { render, screen, fireEvent } from '@testing-library/react';
import Terminal from '../../app/components/Terminal';

describe('Terminal', () => {
  const apiHost = 'http://localhost:5001';

  it('does not render when isOpen is false', () => {
    const { container } = render(<Terminal isOpen={false} onClose={jest.fn()} apiHost={apiHost} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders centered modal when isOpen is true', () => {
    render(<Terminal isOpen={true} onClose={jest.fn()} apiHost={apiHost} />);
    expect(screen.getByText('Trading Cycle Console')).toBeInTheDocument();
  });

  it('shows waiting message when no logs', () => {
    render(<Terminal isOpen={true} onClose={jest.fn()} apiHost={apiHost} />);
    expect(screen.getByText('Waiting for logs...')).toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', () => {
    const onClose = jest.fn();
    render(<Terminal isOpen={true} onClose={onClose} apiHost={apiHost} />);
    const closeBtn = screen.getByTitle('Close');
    fireEvent.click(closeBtn);
    expect(onClose).toHaveBeenCalled();
  });

  it('has a clear button', () => {
    render(<Terminal isOpen={true} onClose={jest.fn()} apiHost={apiHost} />);
    const clearBtn = screen.getByTitle('Clear');
    expect(clearBtn).toBeInTheDocument();
  });

  it('shows line count', () => {
    render(<Terminal isOpen={true} onClose={jest.fn()} apiHost={apiHost} />);
    expect(screen.getByText('0 lines')).toBeInTheDocument();
  });
});
