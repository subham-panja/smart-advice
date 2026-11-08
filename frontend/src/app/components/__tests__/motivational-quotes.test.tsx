import React from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import MotivationalQuotes from '../MotivationalQuotes';

describe('MotivationalQuotes Component', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    act(() => {
      jest.clearAllTimers();
    });
    jest.useRealTimers();
  });

  it('renders without crashing and displays a quote', () => {
    render(<MotivationalQuotes />);
    expect(screen.getByText(/Focus & Anti-Procrastination/i)).toBeInTheDocument();
    expect(screen.getByTitle(/Shuffle Random Quote/i)).toBeInTheDocument();
  });

  it('allows manual quote navigation with Next button', () => {
    render(<MotivationalQuotes />);
    const nextBtn = screen.getByTitle(/Shuffle Random Quote/i);
    expect(nextBtn).toBeInTheDocument();

    act(() => {
      fireEvent.click(nextBtn);
      jest.advanceTimersByTime(300);
    });

    expect(screen.getByText(/Focus & Anti-Procrastination/i)).toBeInTheDocument();
  });

  it('toggles pause and resume auto-refresh', () => {
    render(<MotivationalQuotes />);
    const pauseBtn = screen.getByTitle(/Pause auto-refresh/i);
    expect(pauseBtn).toBeInTheDocument();

    act(() => {
      fireEvent.click(pauseBtn);
    });

    expect(screen.getByTitle(/Resume auto-refresh/i)).toBeInTheDocument();
  });
});
