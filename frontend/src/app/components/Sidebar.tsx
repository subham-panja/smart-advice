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
  XMarkIcon,
} from '@heroicons/react/24/outline';

function SidebarContent() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const currentView = searchParams ? searchParams.get('view') : null;
  const { isCollapsed, toggleSidebar, isMobileOpen, closeMobileSidebar } = useSidebar();
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
      badge: 'SWING',
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
      badge: 'LIVE',
      isLiveBadge: true,
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
    <>
      {/* Mobile Drawer Backdrop Overlay */}
      <div
        className={`fixed inset-0 bg-black/60 backdrop-blur-sm z-40 transition-opacity duration-300 md:hidden ${
          isMobileOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
        onClick={closeMobileSidebar}
        aria-hidden="true"
      />

      {/* Sidebar Container */}
      <aside
        className={`fixed left-0 top-0 h-full bg-white/95 dark:bg-gray-900/95 backdrop-blur-xl border-r border-gray-200/80 dark:border-gray-800/80 shadow-2xl md:shadow-md transition-all duration-300 z-50 md:z-40 flex flex-col ${
          // Mobile: slide drawer in/out
          isMobileOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        } ${
          // Width: on mobile w-72 max-w-[85vw], on desktop collapse/expand
          isCollapsed ? 'w-72 md:w-16' : 'w-72 md:w-64'
        }`}
      >
        {/* Header with Logo, Theme Toggle and Collapse/Close Button */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200/70 dark:border-gray-800/70 h-16">
          {/* Logo & Brand */}
          <Link
            href="/"
            onClick={closeMobileSidebar}
            className="flex items-center space-x-2.5 overflow-hidden group"
          >
            <div className="p-1.5 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 text-white shadow-md shadow-blue-500/20 group-hover:scale-105 transition-transform flex-shrink-0">
              <ChartBarIcon className="h-5 w-5" />
            </div>
            {(!isCollapsed || isMobileOpen) && (
              <div className="truncate">
                <span className="text-base font-bold bg-gradient-to-r from-gray-900 via-gray-800 to-gray-700 dark:from-white dark:via-gray-100 dark:to-gray-300 bg-clip-text text-transparent">
                  Stock Advisor
                </span>
              </div>
            )}
          </Link>

          {/* Action buttons (ThemeToggle + Close/Collapse) */}
          <div className="flex items-center space-x-1">
            <div className={`${isCollapsed && !isMobileOpen ? 'hidden' : 'block'}`}>
              <ThemeToggle collapsed={isCollapsed && !isMobileOpen} />
            </div>

            {/* Desktop Collapse / Expand Button */}
            <button
              onClick={toggleSidebar}
              className="hidden md:flex p-1.5 rounded-lg text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {isCollapsed ? (
                <ChevronRightIcon className="h-5 w-5" />
              ) : (
                <ChevronLeftIcon className="h-5 w-5" />
              )}
            </button>

            {/* Mobile Close Button */}
            <button
              onClick={closeMobileSidebar}
              className="flex md:hidden p-1.5 rounded-lg text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              title="Close navigation drawer"
              aria-label="Close navigation drawer"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 px-3 py-3 space-y-1.5 overflow-y-auto overflow-x-hidden">
          {(!isCollapsed || isMobileOpen) && (
            <div className="px-3 pt-1 pb-1 text-[10px] font-bold uppercase tracking-wider text-gray-400 dark:text-gray-500">
              Stock Analysis
            </div>
          )}

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

            const isDesktopCollapsed = isCollapsed && !isMobileOpen;

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
                    className={`relative flex items-center px-3 py-2 rounded-xl text-sm font-medium transition-all group cursor-pointer ${
                      isDesktopCollapsed ? 'justify-center' : ''
                    } ${
                      isSubChildActive
                        ? 'text-blue-600 dark:text-blue-400 bg-blue-50/70 dark:bg-blue-900/25 font-semibold'
                        : 'text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100/80 dark:hover:bg-gray-800/60'
                    }`}
                    title={isDesktopCollapsed ? item.name : ''}
                  >
                    {isSubChildActive && !isDesktopCollapsed && (
                      <span className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r bg-blue-600 dark:bg-blue-400 shadow-sm shadow-blue-500/50" />
                    )}

                    <Icon
                      className={`h-5 w-5 flex-shrink-0 transition-colors ${
                        isDesktopCollapsed ? '' : 'mr-3'
                      } ${
                        isSubChildActive
                          ? 'text-blue-600 dark:text-blue-400'
                          : 'text-gray-500 dark:text-gray-400 group-hover:text-gray-700 dark:group-hover:text-gray-200'
                      }`}
                    />

                    {!isDesktopCollapsed && (
                      <div className="flex-1 flex items-center justify-between min-w-0 pr-1">
                        <span className="font-semibold text-xs uppercase tracking-wider text-gray-600 dark:text-gray-300 truncate">
                          {item.name}
                        </span>
                        {item.badge && (
                          <span
                            className={`ml-2 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider rounded-md border flex items-center gap-1 ${
                              item.isLiveBadge
                                ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20'
                                : 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20'
                            }`}
                          >
                            {item.isLiveBadge && (
                              <span className="relative flex h-1.5 w-1.5">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
                              </span>
                            )}
                            {item.badge}
                          </span>
                        )}
                      </div>
                    )}

                    {!isDesktopCollapsed && (
                      <div className="ml-1 text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-200">
                        {isExpanded ? (
                          <ChevronUpIcon className="h-4 w-4" />
                        ) : (
                          <ChevronDownIcon className="h-4 w-4" />
                        )}
                      </div>
                    )}
                  </div>

                  {isExpanded && !isDesktopCollapsed && (
                    <ul className="pl-3.5 space-y-1 border-l border-gray-200/80 dark:border-gray-800 ml-4">
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
                              onClick={closeMobileSidebar}
                              className={`flex items-center px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all group ${
                                isSubActive
                                  ? 'bg-blue-100/90 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 font-semibold shadow-sm'
                                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100/70 dark:hover:bg-gray-800/40'
                              }`}
                            >
                              <SubIcon
                                className={`h-3.5 w-3.5 flex-shrink-0 mr-2 ${
                                  isSubActive
                                    ? 'text-blue-600 dark:text-blue-400'
                                    : 'text-gray-400 group-hover:text-gray-600 dark:group-hover:text-gray-300'
                                }`}
                              />
                              <span className="truncate">{subItem.name}</span>
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
                  onClick={closeMobileSidebar}
                  className={`relative flex items-center px-3 py-2 rounded-xl text-sm font-medium transition-all group ${
                    isDesktopCollapsed ? 'justify-center' : ''
                  } ${
                    isActive
                      ? 'bg-gradient-to-r from-blue-500/10 to-indigo-500/10 text-blue-700 dark:text-blue-300 font-semibold shadow-sm'
                      : 'text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100/80 dark:hover:bg-gray-800/60'
                  }`}
                  title={isDesktopCollapsed ? item.name : ''}
                >
                  {isActive && !isDesktopCollapsed && (
                    <span className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r bg-blue-600 dark:bg-blue-400 shadow-sm shadow-blue-500/50" />
                  )}

                  <Icon
                    className={`h-5 w-5 flex-shrink-0 transition-colors ${
                      isDesktopCollapsed ? '' : 'mr-3'
                    } ${
                      isActive
                        ? 'text-blue-600 dark:text-blue-400'
                        : 'text-gray-500 dark:text-gray-400 group-hover:text-gray-700 dark:group-hover:text-gray-200'
                    }`}
                  />
                  {!isDesktopCollapsed && (
                    <div className="truncate">
                      <div className="font-medium text-xs leading-none">{item.name}</div>
                      <div className="text-[11px] text-gray-500 dark:text-gray-400 mt-0.5 truncate">
                        {item.description}
                      </div>
                    </div>
                  )}
                </Link>
              );
            }
          })}
        </nav>

        {/* Bottom Section with Settings and About */}
        <div className="mt-auto border-t border-gray-200/70 dark:border-gray-800/70">
          <div className="p-2.5">
            <Link
              href={settingsItem.href}
              onClick={closeMobileSidebar}
              className={`flex items-center px-3 py-2 rounded-xl text-sm font-medium transition-colors group ${
                isCollapsed && !isMobileOpen ? 'justify-center' : ''
              } ${
                pathname === settingsItem.href
                  ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 font-semibold'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100/80 dark:hover:bg-gray-800/60'
              }`}
              title={isCollapsed && !isMobileOpen ? settingsItem.name : ''}
            >
              <CogIcon
                className={`h-5 w-5 flex-shrink-0 ${
                  isCollapsed && !isMobileOpen ? '' : 'mr-3'
                } text-gray-500 dark:text-gray-400 group-hover:text-gray-700 dark:group-hover:text-gray-200`}
              />
              {(!isCollapsed || isMobileOpen) && (
                <div className="truncate">
                  <div className="font-medium text-xs">{settingsItem.name}</div>
                  <div className="text-[11px] text-gray-500 dark:text-gray-400">
                    {settingsItem.description}
                  </div>
                </div>
              )}
            </Link>
          </div>

          {/* Footer Info */}
          <div className="px-4 py-3 border-t border-gray-100 dark:border-gray-800/50 flex items-center justify-between">
            {(!isCollapsed || isMobileOpen) && (
              <span className="text-[11px] font-medium text-gray-400 dark:text-gray-500">
                Stock Advisor v1.0
              </span>
            )}
            <Link
              href="/about"
              onClick={closeMobileSidebar}
              className="p-1 rounded-md text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              title="About"
            >
              <InformationCircleIcon className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </aside>
    </>
  );
}

export default function Sidebar() {
  return (
    <Suspense fallback={<div className="w-64 bg-white dark:bg-gray-800 h-full fixed hidden md:block" />}>
      <SidebarContent />
    </Suspense>
  );
}
