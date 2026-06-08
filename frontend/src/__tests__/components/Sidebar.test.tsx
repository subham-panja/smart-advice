import { render, screen } from '@testing-library/react';
import Sidebar from '../../app/components/Sidebar';
import { SidebarProvider } from '../../app/contexts/SidebarContext';
import { ThemeProvider } from '../../app/contexts/ThemeContext';

const renderWithProviders = () => {
  return render(
    <ThemeProvider>
      <SidebarProvider>
        <Sidebar />
      </SidebarProvider>
    </ThemeProvider>
  );
};

describe('Sidebar', () => {
  it('renders Dashboard link', () => {
    renderWithProviders();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });

  it('renders Stock Analysis section', () => {
    renderWithProviders();
    expect(screen.getByText('Stock Analysis')).toBeInTheDocument();
  });

  it('renders Positions link', () => {
    renderWithProviders();
    expect(screen.getByText('Positions')).toBeInTheDocument();
  });

  it('renders Activity Log link', () => {
    renderWithProviders();
    expect(screen.getByText('Activity Log')).toBeInTheDocument();
  });

  it('renders Settings link', () => {
    renderWithProviders();
    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('does not render F&O Analysis (disabled)', () => {
    renderWithProviders();
    expect(screen.queryByText('F&O Analysis')).not.toBeInTheDocument();
  });

  it('renders app title', () => {
    renderWithProviders();
    expect(screen.getByText('Stock Advisor')).toBeInTheDocument();
  });

  it('renders version text', () => {
    renderWithProviders();
    expect(screen.getByText('Stock Advisor v1.0')).toBeInTheDocument();
  });
});
