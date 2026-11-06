'use client';

import Link from 'next/link';
import { usePathname, useSearchParams } from 'next/navigation';
import { useState, Suspense } from 'react';
import { useSidebar } from '../contexts/SidebarContext';
import ThemeToggle from './ThemeToggle';
import {
  ChartBarIcon,
  PlayIcon,
  CogIcon,
  HomeIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  InformationCircleIcon,
  ClockIcon,
  DocumentMagnifyingGlassIcon,
  BoltIcon,
  ArrowTrendingUpIcon,
  BriefcaseIcon,
  FireIcon,
} from '@heroicons/react/24/outline';

function SidebarContent() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const currentView = searchParams.get('view');
  const { isCollapsed, toggleSidebar } = useSidebar();
  const [expandedMenus, setExpandedMenus] = useState<string[]>([
    'Swing Trading',
    'Intraday Trading',
  ]);

  const navigationItems = [
    {
      name: 'Dashboard',
      href: '/',
      icon: HomeIcon,
      description: 'Overview & metrics',
    },
    {
      name: 'Swing Trading',
      icon: ArrowTrendingUpIcon,
      description: 'Daily analysis & holding',
      submenu: [
        {
          name: 'Trading Cycle',
          href: '/analysis',
          icon: PlayIcon,
          description: 'Run swing analysis cycle',
        },
        {
          name: 'Stock Recommendations',
          href: '/recommendations',
          icon: ChartBarIcon,
          description: 'Qualified swing stock setups',
        },
        {
          name: 'Positions',
          href: '/positions',
          icon: BriefcaseIcon,
          description: 'Manage open swing positions',
        },
      ],
    },
    {
      name: 'Intraday Trading',
      icon: BoltIcon,
      description: 'ORB, VWAP & CPR intraday',
      submenu: [
        {
          name: 'Live Cockpit & Cycle',
          href: '/intraday',
          icon: FireIcon,
          description: 'Intraday status & square-off',
          matchView: null,
        },
        {
          name: 'Recommendations',
          href: '/intraday?view=recommendations',
          icon: ChartBarIcon,
          description: 'Breakout & confluence setups',
          matchView: 'recommendations',
        },
        {
          name: 'Active Positions',
          href: '/intraday?view=positions',
          icon: BriefcaseIcon,
          description: '15:15 IST active trades',
          matchView: 'positions',
        },
      ],
    },
    {
      name: 'Backtests',
      href: '/backtests',
      icon: DocumentMagnifyingGlassIcon,
      description: 'Walk-forward & MC sessions',
    },
    {
      name: 'Activity Log',
      href: '/activity-log',
      icon: ClockIcon,
      description: 'Audit logs & trade events',
    },
  ];

  const settingsItem = {
    name: 'Settings',
    href: '/settings',
    icon: CogIcon,
    description: 'Configure preferences',
  };

  return (
    <div
      className={`fixed left-0 top-0 h-full bg-white dark:bg-gray-800 shadow-lg border-r border-gray-200 dark:border-gray-700 transition-all duration-300 z-40 flex flex-col ${
        isCollapsed ? 'w-16' : 'w-64'
      }`}
    >
      {/* Header with Logo, Theme Toggle and Collapse Toggle */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        {!isCollapsed && (
          <Link href="/" className="flex items-center space-x-2">
            <ChartBarIcon className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            <span className="text-xl font-bold text-gray-900 dark:text-gray-100">
              Stock Advisor
            </span>
          </Link>
        )}
        {isCollapsed && (
          <Link href="/" className="flex items-center justify-center w-8">
            <ChartBarIcon className="h-8 w-8 text-blue-600 dark:text-blue-400" />
          </Link>
        )}
        <div className="flex items-center space-x-2">
          <ThemeToggle collapsed={isCollapsed} />
          <button
            onClick={toggleSidebar}
            className="p-1.5 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {isCollapsed ? (
              <ChevronRightIcon className="h-5 w-5" />
            ) : (
              <ChevronLeftIcon className="h-5 w-5" />
            )}
          </button>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-3 space-y-1.5 overflow-y-auto">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const hasSubmenu = Array.isArray(item.submenu);
          const isExpanded = expandedMenus.includes(item.name);
          const toggleExpand = () =>
            setExpandedMenus((prev) =>
              prev.includes(item.name)
                ? prev.filter((name) => name !== item.name)
                : [...prev, item.name]
            );

          if (hasSubmenu) {
            const isSubChildActive = item.submenu?.some((sub: any) => {
              if (pathname === '/intraday' && sub.href.startsWith('/intraday')) {
                if (sub.matchView) return currentView === sub.matchView;
                return !currentView;
              }
              return pathname === sub.href;
            });

            return (
              <div key={item.name} className="space-y-1">
                <div
                  onClick={toggleExpand}
                  className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors group cursor-pointer ${
                    isCollapsed ? 'justify-center' : ''
                  } ${
                    isSubChildActive
                      ? 'text-blue-600 dark:text-blue-400 bg-blue-50/50 dark:bg-blue-900/20'
                      : 'text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700'
                  }`}
                  title={isCollapsed ? item.name : ''}
                >
                  <Icon
                    className={`h-5 w-5 flex-shrink-0 ${
                      isCollapsed ? '' : 'mr-3'
                    } ${isSubChildActive ? 'text-blue-600 dark:text-blue-400' : ''}`}
                  />
                  {!isCollapsed && (
                    <div className="flex-1">
                      <div className="font-semibold text-xs uppercase tracking-wider text-gray-500 dark:text-gray-400">
                        {item.name}
                      </div>
                    </div>
                  )}
                  {!isCollapsed && (
                    <div className="ml-auto text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-200">
                      {isExpanded ? (
                        <ChevronUpIcon className="h-4 w-4" />
                      ) : (
                        <ChevronDownIcon className="h-4 w-4" />
                      )}
                    </div>
                  )}
                </div>

                {isExpanded && !isCollapsed && (
                  <ul className="pl-4 space-y-1 border-l-2 border-gray-100 dark:border-gray-700/60 ml-4">
                    {item.submenu?.map((subItem: any) => {
                      const SubIcon = subItem.icon;
                      let isSubActive = false;
                      if (pathname === '/intraday' && subItem.href.startsWith('/intraday')) {
                        isSubActive = subItem.matchView ? currentView === subItem.matchView : !currentView;
                      } else {
                        isSubActive = pathname === subItem.href;
                      }

                      return (
                        <li key={subItem.name}>
                          <Link
                            href={subItem.href}
                            className={`flex items-center px-2.5 py-1.5 rounded-md text-xs font-medium transition-colors group ${
                              isSubActive
                                ? 'bg-blue-100 dark:bg-blue-900/60 text-blue-700 dark:text-blue-300 font-semibold'
                                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700/50'
                            }`}
                          >
                            <SubIcon className="h-3.5 w-3.5 flex-shrink-0 mr-2" />
                            <div className="truncate">{subItem.name}</div>
                          </Link>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            );
          } else {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href!}
                className={`flex items-center px-3 py-2 rounded-lg text-sm font-medium transition-colors group ${
                  isCollapsed ? 'justify-center' : ''
                } ${
                  isActive
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                    : 'text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                title={isCollapsed ? item.name : ''}
              >
                <Icon
                  className={`h-5 w-5 flex-shrink-0 ${isCollapsed ? '' : 'mr-3'}`}
                />
                {!isCollapsed && (
                  <div>
                    <div className="font-medium text-xs">{item.name}</div>
                    <div className="text-[11px] text-gray-500 dark:text-gray-400">
                      {item.description}
                    </div>
                  </div>
                )}
              </Link>
            );
          }
        })}
      </nav>

      {/* Bottom Section with Settings */}
      <div className="mt-auto">
        <div className="p-3 border-t border-gray-200 dark:border-gray-700">
          <Link
            href={settingsItem.href}
            className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors group ${
              isCollapsed ? 'justify-center' : ''
            } ${
              pathname === settingsItem.href
                ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            title={isCollapsed ? settingsItem.name : ''}
          >
            <CogIcon
              className={`h-5 w-5 flex-shrink-0 ${isCollapsed ? '' : 'mr-3'}`}
            />
            {!isCollapsed && (
              <div>
                <div className="font-medium text-xs">{settingsItem.name}</div>
                <div className="text-[11px] text-gray-500 dark:text-gray-400">
                  {settingsItem.description}
                </div>
              </div>
            )}
          </Link>
        </div>

        {/* Footer with About */}
        <div className="p-3 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            {!isCollapsed && (
              <span className="text-[11px] text-gray-500 dark:text-gray-400">
                Stock Advisor v1.0
              </span>
            )}
            <Link
              href="/about"
              className="p-1.5 rounded-md text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="About"
            >
              <InformationCircleIcon className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Sidebar() {
  return (
    <Suspense fallback={<div className="w-64 bg-white dark:bg-gray-800 h-full fixed" />}>
      <SidebarContent />
    </Suspense>
  );
}
