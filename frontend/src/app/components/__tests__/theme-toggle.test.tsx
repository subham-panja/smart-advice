'use client';

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ThemeProvider, useTheme } from '../../contexts/ThemeContext';
import '@testing-library/jest-dom';

// Mock localStorage
const localStorageMock = {
  getItem: jest.fn(),
  setItem: jest.fn(),
  removeItem: jest.fn(),
  clear: jest.fn(),
};

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
});

// Mock matchMedia
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

// Test component that uses the theme context
function TestComponent() {
  const { theme, toggleTheme, setTheme, mounted } = useTheme();
  
  if (!mounted) {
    return <div data-testid="loading">Loading...</div>;
  }
  
  return (
    <div data-testid="test-component">
      <div data-testid="current-theme">{theme}</div>
      <button data-testid="toggle-theme" onClick={toggleTheme}>
        Toggle Theme
      </button>
      <button data-testid="set-light" onClick={() => setTheme('light')}>
        Set Light
      </button>
      <button data-testid="set-dark" onClick={() => setTheme('dark')}>
        Set Dark
      </button>
      <button data-testid="set-system" onClick={() => setTheme('system')}>
        Set System
      </button>
    </div>
  );
}

describe('Theme Context and Toggle', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
    localStorageMock.getItem.mockReturnValue(null);
    
    // Clear document classes
    document.documentElement.className = '';
    document.documentElement.removeAttribute('data-theme');
  });

  it('should initialize with light theme by default', async () => {
    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });

    expect(document.documentElement).not.toHaveClass('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('should load theme from localStorage', async () => {
    localStorageMock.getItem.mockReturnValue('dark');

    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('dark');
    });

    expect(document.documentElement).toHaveClass('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('should toggle between light and dark themes', async () => {
    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    // Wait for initial load
    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });

    // Toggle to dark
    fireEvent.click(screen.getByTestId('toggle-theme'));

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('dark');
    });

    expect(document.documentElement).toHaveClass('dark');
    expect(localStorageMock.setItem).toHaveBeenCalledWith('theme', 'dark');

    // Toggle back to light
    fireEvent.click(screen.getByTestId('toggle-theme'));

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });

    expect(document.documentElement).not.toHaveClass('dark');
    expect(localStorageMock.setItem).toHaveBeenCalledWith('theme', 'light');
  });

  it('should set specific themes correctly', async () => {
    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });

    // Set dark theme
    fireEvent.click(screen.getByTestId('set-dark'));

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('dark');
    });

    expect(document.documentElement).toHaveClass('dark');

    // Set light theme
    fireEvent.click(screen.getByTestId('set-light'));

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });

    expect(document.documentElement).not.toHaveClass('dark');
  });

  it('should handle system theme correctly', async () => {
    // Mock system preference for dark mode
    window.matchMedia = jest.fn().mockImplementation(query => ({
      matches: query === '(prefers-color-scheme: dark)',
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });

    // Set system theme
    fireEvent.click(screen.getByTestId('set-system'));

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('system');
    });

    // Should apply dark theme based on system preference
    expect(document.documentElement).toHaveClass('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('should handle system theme with light preference', async () => {
    // Mock system preference for light mode
    window.matchMedia = jest.fn().mockImplementation(query => ({
      matches: false, // light mode
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));

    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });

    // Set system theme
    fireEvent.click(screen.getByTestId('set-system'));

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('system');
    });

    // Should apply light theme based on system preference
    expect(document.documentElement).not.toHaveClass('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
  });

  it('should save theme to localStorage when changed', async () => {
    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });

    // Change to dark theme
    fireEvent.click(screen.getByTestId('set-dark'));

    await waitFor(() => {
      expect(localStorageMock.setItem).toHaveBeenCalledWith('theme', 'dark');
    });

    // Change to system theme
    fireEvent.click(screen.getByTestId('set-system'));

    await waitFor(() => {
      expect(localStorageMock.setItem).toHaveBeenCalledWith('theme', 'system');
    });
  });

  it('should handle localStorage errors gracefully', async () => {
    const consoleSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    localStorageMock.getItem.mockImplementation(() => {
      throw new Error('localStorage not available');
    });

    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });

    expect(consoleSpy).toHaveBeenCalledWith(
      'Error accessing localStorage or window:',
      expect.any(Error)
    );

    consoleSpy.mockRestore();
  });

  it('should handle invalid localStorage values', async () => {
    localStorageMock.getItem.mockReturnValue('invalid-theme');

    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    await waitFor(() => {
      // Should fall back to default light theme
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });
  });

  it('should update DOM classes correctly during theme transitions', async () => {
    render(
      <ThemeProvider>
        <TestComponent />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByTestId('current-theme')).toHaveTextContent('light');
    });

    // Initially should not have dark class
    expect(document.documentElement).not.toHaveClass('dark');

    // Switch to dark
    fireEvent.click(screen.getByTestId('set-dark'));

    await waitFor(() => {
      expect(document.documentElement).toHaveClass('dark');
    });

    // Switch back to light
    fireEvent.click(screen.getByTestId('set-light'));

    await waitFor(() => {
      expect(document.documentElement).not.toHaveClass('dark');
    });
  });
});
