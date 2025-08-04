'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { useSidebar } from '../contexts/SidebarContext';
import { 
  ChartBarIcon, 
  PlayIcon, 
  CogIcon, 
  Bars3Icon,
  XMarkIcon,
  HomeIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  InformationCircleIcon
} from '@heroicons/react/24/outline';

const Sidebar = () => {
  const pathname = usePathname();
  const { isCollapsed, toggleSidebar } = useSidebar();
  const [expandedMenus, setExpandedMenus] = useState<string[]>([]);

  const navigationItems = [
    {
      name: 'Dashboard',
      href: '/',
      icon: HomeIcon,
      description: 'Main dashboard'
    },
    {
      name: 'Stock Analysis',
      icon: ChartBarIcon,
      description: 'Stock market analysis tools',
      submenu: [
        {
          name: 'Generate Analysis',
          href: '/analysis',
          icon: PlayIcon,
          description: 'Create new stock analysis'
        },
        {
          name: 'View Recommendations',
          href: '/recommendations',
          icon: ChartBarIcon,
          description: 'Browse stock recommendations'
        }
      ]
    },
    {
      name: 'F&O Analysis',
      icon: PlayIcon,
      description: 'Futures & Options analysis',
      submenu: [
        {
          name: 'F&O Analysis',
          href: '/fo-analysis',
          icon: PlayIcon,
          description: 'Create F&O analysis'
        },
        {
          name: 'View F&O Recommendations',
          href: '/fo-recommendations',
          icon: ChartBarIcon,
          description: 'Browse F&O recommendations'
        }
      ]
    },
  ];

  const settingsItem = {
    name: 'Settings',
    href: '/settings',
    icon: CogIcon,
    description: 'Configure preferences'
  };

  return (
    <div className={`fixed left-0 top-0 h-full bg-white dark:bg-gray-800 shadow-lg border-r border-gray-200 dark:border-gray-700 transition-all duration-300 z-40 ${
      isCollapsed ? 'w-16' : 'w-64'
    }`}>
      {/* Header with Logo and Toggle */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        {!isCollapsed && (
          <Link href="/" className="flex items-center space-x-2">
            <ChartBarIcon className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            <span className="text-xl font-bold text-gray-900 dark:text-gray-100">Stock Advisor</span>
          </Link>
        )}
        {isCollapsed && (
          <Link href="/" className="flex items-center justify-center w-8">
            <ChartBarIcon className="h-8 w-8 text-blue-600 dark:text-blue-400" />
          </Link>
        )}
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

      {/* Navigation Links */}
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {navigationItems.map((item) => {
          const Icon = item.icon;
          const hasSubmenu = Array.isArray(item.submenu);
          const isExpanded = expandedMenus.includes(item.name);
          const toggleExpand = () =>
            setExpandedMenus(prev =>
              prev.includes(item.name)
                ? prev.filter(name => name !== item.name)
                : [...prev, item.name]
            );

          if (hasSubmenu) {
            return (
              <div key={item.name}>
                <div
                  onClick={toggleExpand}
                  className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors group cursor-pointer ${
                    isCollapsed ? 'justify-center' : ''
                  } text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700`}
                  title={isCollapsed ? item.name : ''}
                >
                  <Icon className={`h-5 w-5 flex-shrink-0 ${isCollapsed ? '' : 'mr-3'}`} />
                  {!isCollapsed && (
                    <div className="flex-1">
                      <div className="font-medium">{item.name}</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400">
                        {item.description}
                      </div>
                    </div>
                  )}
                  {!isCollapsed && (
                    <div className="ml-auto">
                      {isExpanded ? <ChevronUpIcon className="h-4 w-4" /> : <ChevronDownIcon className="h-4 w-4" />}
                    </div>
                  )}
                </div>
                {isExpanded && !isCollapsed && (
                  <ul className="pl-8 mt-1 space-y-1">
                    {item.submenu.map((subItem) => {
                      const SubIcon = subItem.icon;
                      const isSubActive = pathname === subItem.href;
                      return (
                        <li key={subItem.name}>
                          <Link
                            href={subItem.href}
                            className={`flex items-center px-3 py-1 rounded-md text-sm font-medium transition-colors group ${
                              isSubActive
                                ? 'bg-blue-50 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300'
                                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700'
                            }`}
                          >
                            <SubIcon className="h-4 w-4 flex-shrink-0 mr-2" />
                            <div>{subItem.name}</div>
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
                href={item.href}
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors group ${
                  isCollapsed ? 'justify-center' : ''
                } ${
                  isActive
                    ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                    : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                title={isCollapsed ? item.name : ''}
              >
                <Icon className={`h-5 w-5 flex-shrink-0 ${isCollapsed ? '' : 'mr-3'}`} />
                {!isCollapsed && (
                  <div>
                    <div className="font-medium">{item.name}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {item.description}
                    </div>
                  </div>
                )}
              </Link>
            );
          }
        })}
      </nav>

      {/* Bottom Section with About and Settings */}
      <div className="mt-auto">
        {/* About Link */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <Link
            href="/about"
            className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors group ${
              isCollapsed ? 'justify-center' : ''
            } ${
              pathname === '/about'
                ? 'bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            title={isCollapsed ? 'About' : ''}
          >
            <InformationCircleIcon className={`h-5 w-5 flex-shrink-0 ${isCollapsed ? '' : 'mr-3'}`} />
            {!isCollapsed && (
              <div>
                <div className="font-medium">About</div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  Learn about the platform
                </div>
              </div>
            )}
          </Link>
        </div>
        
        {/* Settings Link */}
        <div className="px-4 pb-4">
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
            <CogIcon className={`h-5 w-5 flex-shrink-0 ${isCollapsed ? '' : 'mr-3'}`} />
            {!isCollapsed && (
              <div>
                <div className="font-medium">{settingsItem.name}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {settingsItem.description}
                </div>
              </div>
            )}
          </Link>
        </div>
        
        {/* Footer */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          <div className="text-center text-xs text-gray-500 dark:text-gray-400">
            {!isCollapsed && 'Stock Advisor v1.0'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
