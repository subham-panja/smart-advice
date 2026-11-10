'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import {
  ChartBarIcon,
  PlayIcon,
  CogIcon,
  ChevronDownIcon,
  Bars3Icon,
} from '@heroicons/react/24/outline';
import ThemeToggle from './ThemeToggle';
import { useSidebar } from '../contexts/SidebarContext';

const Navbar = () => {
  const pathname = usePathname();
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const { toggleMobileSidebar } = useSidebar();

  const stockAnalysisItems = [
    {
      name: 'Generate Analysis',
      href: '/analysis',
      icon: PlayIcon,
      description: 'Create new stock analysis',
    },
    {
      name: 'View Recommendations',
      href: '/recommendations',
      icon: ChartBarIcon,
      description: 'Browse current recommendations',
    },
    {
      name: 'Settings',
      href: '/settings',
      icon: CogIcon,
      description: 'Configure preferences',
    },
  ];

  const isStockAnalysisActive = stockAnalysisItems.some((item) => pathname === item.href);

  return (
    <nav className="sticky top-0 z-30 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200/70 dark:border-gray-800/70 transition-colors w-full">
      <div className="w-full px-3 sm:px-6">
        <div className="flex justify-between items-center h-16 gap-2">
          {/* Left: Mobile hamburger menu & Brand Logo */}
          <div className="flex items-center space-x-2 sm:space-x-3 min-w-0">
            {/* Hamburger button for mobile */}
            <button
              type="button"
              onClick={toggleMobileSidebar}
              className="p-2 -ml-1 rounded-lg text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors md:hidden focus:outline-none focus:ring-2 focus:ring-blue-500"
              aria-label="Open mobile navigation menu"
            >
              <Bars3Icon className="h-6 w-6" />
            </button>

            {/* Logo */}
            <Link
              href="/"
              className="flex items-center space-x-2 flex-shrink-0 group"
            >
              <div className="p-1.5 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 text-white shadow-sm group-hover:scale-105 transition-transform">
                <ChartBarIcon className="h-5 w-5" />
              </div>
              <span className="text-base sm:text-lg font-bold text-gray-900 dark:text-gray-100 tracking-tight">
                Stock Advisor
              </span>
            </Link>

            {/* Live Market Status Pill */}
            <div className="hidden xs:flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-medium ml-1 sm:ml-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="font-semibold tracking-wide">NSE</span>
              <span className="hidden sm:inline text-[11px] opacity-80">• Active Market</span>
            </div>
          </div>

          {/* Right: Stock Analysis Dropdown & Theme Toggle */}
          <div className="flex items-center space-x-2 sm:space-x-4 flex-shrink-0">
            <div className="relative">
              {/* Stock Analysis Dropdown Button */}
              <button
                type="button"
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                onBlur={() => setTimeout(() => setIsDropdownOpen(false), 200)}
                className={`flex items-center space-x-1.5 sm:space-x-2 px-2.5 sm:px-3 py-1.5 sm:py-2 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
                  isStockAnalysisActive
                    ? 'bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300'
                    : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
              >
                <ChartBarIcon className="h-4 w-4 sm:h-5 sm:w-5 flex-shrink-0" />
                <span className="truncate">Stock Analysis</span>
                <ChevronDownIcon
                  className={`h-4 w-4 transition-transform duration-200 ${
                    isDropdownOpen ? 'rotate-180' : ''
                  }`}
                />
              </button>

              {/* Dropdown Menu */}
              {isDropdownOpen && (
                <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-200 dark:border-gray-700 z-50 py-1 overflow-hidden">
                  {stockAnalysisItems.map((item) => {
                    const Icon = item.icon;
                    const isActive = pathname === item.href;
                    return (
                      <Link
                        key={item.name}
                        href={item.href}
                        className={`flex items-center px-4 py-3 text-sm hover:bg-gray-50 dark:hover:bg-gray-700/60 transition-colors ${
                          isActive
                            ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                            : 'text-gray-700 dark:text-gray-300'
                        }`}
                        onClick={() => setIsDropdownOpen(false)}
                      >
                        <Icon className="h-5 w-5 mr-3 flex-shrink-0 text-gray-500 dark:text-gray-400" />
                        <div>
                          <div className="font-medium text-xs sm:text-sm">{item.name}</div>
                          <div className="text-[11px] text-gray-500 dark:text-gray-400">
                            {item.description}
                          </div>
                        </div>
                      </Link>
                    );
                  })}
                </div>
              )}
            </div>

            <ThemeToggle />
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
