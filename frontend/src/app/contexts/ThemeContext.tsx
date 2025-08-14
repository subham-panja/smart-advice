'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';

type Theme = 'light' | 'dark' | 'system';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
  mounted: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

function getSystemTheme(): 'light' | 'dark' {
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  
  let actualTheme: 'light' | 'dark';
  if (theme === 'system') {
    actualTheme = getSystemTheme();
  } else {
    actualTheme = theme;
  }
  
  // For Tailwind CSS dark mode, we only need to add 'dark' class for dark mode
  // and remove it for light mode
  if (actualTheme === 'dark') {
    root.classList.add('dark');
  } else {
    root.classList.remove('dark');
  }
  
  // Also set data attribute for additional CSS targeting
  root.setAttribute('data-theme', actualTheme);
  
  console.log('Theme applied:', theme, 'Actual theme:', actualTheme, 'Has dark class:', root.classList.contains('dark'), 'Root classes:', root.classList.toString());
}

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<Theme>('light');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    
    // Get initial theme - default to light instead of system
    let initialTheme: Theme = 'light';
    
    try {
      const stored = localStorage.getItem('theme') as Theme;
      
      if (stored && (stored === 'light' || stored === 'dark' || stored === 'system')) {
        initialTheme = stored;
      }
    } catch (error) {
      // Handle SSR or localStorage access errors
      console.warn('Error accessing localStorage or window:', error);
    }
    
    console.log('Initial theme:', initialTheme);
    setTheme(initialTheme);
    // Apply theme immediately during initialization
    if (typeof window !== 'undefined') {
      applyTheme(initialTheme);
    }
  }, []);

  useEffect(() => {
    if (!mounted) return;
    
    applyTheme(theme);
    localStorage.setItem('theme', theme);
  }, [theme, mounted]);

  // Listen for system theme changes
  useEffect(() => {
    if (!mounted || theme !== 'system') return;

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      if (theme === 'system') {
        applyTheme('system');
      }
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme, mounted]);

  const toggleTheme = () => {
    console.log('Toggle theme called, current:', theme);
    const newTheme = theme === 'light' ? 'dark' : 'light';
    console.log('Setting new theme:', newTheme);
    setTheme(newTheme);
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme, mounted }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
