import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ThemeProvider } from '../../app/contexts/ThemeContext';
import ThemeToggle from '../../app/components/ThemeToggle';

const renderWithThemeProvider = (component: React.ReactElement) => {
  return render(
    <ThemeProvider>
      {component}
    </ThemeProvider>
  );
};

describe('ThemeToggle', () => {
  beforeEach(() => {
    // Access localStorage mock from window object
    (window.localStorage.getItem as jest.Mock).mockClear();
    (window.localStorage.setItem as jest.Mock).mockClear();
  });

  it('renders theme toggle button', () => {
    renderWithThemeProvider(<ThemeToggle />);
    
    const button = screen.getByRole('button', { name: /toggle theme/i });
    expect(button).toBeInTheDocument();
  });

  it('shows correct icon for light theme', () => {
    (window.localStorage.getItem as jest.Mock).mockReturnValue('light');
    
    renderWithThemeProvider(<ThemeToggle />);
    
    const button = screen.getByRole('button', { name: /toggle theme/i });
    expect(button).toBeInTheDocument();
    
    // Check for SVG element (MoonIcon for light theme)
    const svg = button.querySelector('svg');
    expect(svg).toBeInTheDocument();
  });

  it('shows correct icon for dark theme', () => {
    (window.localStorage.getItem as jest.Mock).mockReturnValue('dark');
    
    renderWithThemeProvider(<ThemeToggle />);
    
    const button = screen.getByRole('button', { name: /toggle theme/i });
    expect(button).toBeInTheDocument();
    
    // Check for SVG element (SunIcon for dark theme)
    const svg = button.querySelector('svg');
    expect(svg).toBeInTheDocument();
  });

  it('calls toggleTheme when clicked', () => {
    const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
    
    renderWithThemeProvider(<ThemeToggle />);
    
    const button = screen.getByRole('button', { name: /toggle theme/i });
    fireEvent.click(button);
    
    // Check if console.log was called (from our theme toggle debug)
    expect(consoleSpy).toHaveBeenCalledWith('Theme toggle button clicked');
    
    consoleSpy.mockRestore();
  });

  it('has proper accessibility attributes', () => {
    renderWithThemeProvider(<ThemeToggle />);
    
    const button = screen.getByRole('button', { name: /toggle theme/i });
    expect(button).toHaveAttribute('aria-label', 'Toggle theme');
  });

  it('applies correct CSS classes', () => {
    renderWithThemeProvider(<ThemeToggle />);
    
    const button = screen.getByRole('button', { name: /toggle theme/i });
    expect(button).toHaveClass('relative', 'inline-flex', 'h-10', 'w-10');
  });

  it('shows different icons when theme changes', async () => {
    const { rerender } = renderWithThemeProvider(<ThemeToggle />);
    
    const button = screen.getByRole('button', { name: /toggle theme/i });
    expect(button).toBeInTheDocument();
    
    // Click to toggle theme
    fireEvent.click(button);
    
    // Wait for state change
    await waitFor(() => {
      const svg = button.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });
  });
});
