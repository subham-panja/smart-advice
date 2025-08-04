import { render, screen, fireEvent } from '@testing-library/react';
import Navbar from '../../app/components/Navbar';
import { ThemeProvider } from '../../app/contexts/ThemeContext';

const renderWithThemeProvider = (component: React.ReactElement) => {
  return render(
    <ThemeProvider>
      {component}
    </ThemeProvider>
  );
};

describe('Navbar', () => {
  it('renders navbar with Stock Advisor logo', () => {
    renderWithThemeProvider(<Navbar />);
    const logo = screen.getByText(/Stock Advisor/i);
    expect(logo).toBeInTheDocument();
  });

  it('shows dropdown menu items when Stock Analysis is clicked', () => {
    renderWithThemeProvider(<Navbar />);
    
    // Click on Stock Analysis dropdown button
    const stockAnalysisButton = screen.getByText(/Stock Analysis/i);
    fireEvent.click(stockAnalysisButton);
    
    // Check that dropdown items appear
    const generateAnalysisLink = screen.getByText(/Generate Analysis/i);
    const recommendationsLink = screen.getByText(/View Recommendations/i);
    const settingsLink = screen.getByText(/Settings/i);
    
    expect(generateAnalysisLink).toBeInTheDocument();
    expect(recommendationsLink).toBeInTheDocument();
    expect(settingsLink).toBeInTheDocument();
  });

  it('renders ThemeToggle component', () => {
    renderWithThemeProvider(<Navbar />);
    const themeToggle = screen.getByRole('button', { name: /toggle theme/i });
    expect(themeToggle).toBeInTheDocument();
  });

  it('has Stock Analysis dropdown button', () => {
    renderWithThemeProvider(<Navbar />);
    const stockAnalysisButton = screen.getByText(/Stock Analysis/i).closest('button');
    expect(stockAnalysisButton).toBeInTheDocument();
    expect(stockAnalysisButton).toHaveClass('flex', 'items-center');
  });

  it('contains chevron icon in dropdown button', () => {
    renderWithThemeProvider(<Navbar />);
    const stockAnalysisButton = screen.getByText(/Stock Analysis/i).closest('button');
    const chevronIcon = stockAnalysisButton?.querySelector('svg');
    expect(chevronIcon).toBeInTheDocument();
  });
});

