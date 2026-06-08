'use client';

import { useState, useEffect } from 'react';
import {
  CogIcon,
  SunIcon,
  MoonIcon,
  ComputerDesktopIcon,
  ChartBarIcon,
  ShieldCheckIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { useTheme } from '../contexts/ThemeContext';
import { getTradingConfig, getStrategies, TradingConfig, Strategy } from '@/lib/api';

interface SettingsSection {
  id: string;
  name: string;
  icon: React.ComponentType<{ className?: string }>;
}

const settingsSections: SettingsSection[] = [
  { id: 'general', name: 'General', icon: CogIcon },
  { id: 'trading', name: 'Trading Config', icon: ShieldCheckIcon },
  { id: 'strategies', name: 'Strategies', icon: ChartBarIcon },
];

export default function Settings() {
  const { theme, setTheme } = useTheme();
  const [activeSection, setActiveSection] = useState('general');
  const [tradingConfig, setTradingConfig] = useState<TradingConfig | null>(null);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<Strategy | null>(null);

  useEffect(() => {
    const fetch = async () => {
      const [tc, st] = await Promise.all([getTradingConfig(), getStrategies()]);
      if (tc.status === 'success') setTradingConfig(tc.config);
      if (st.status === 'success') setStrategies(st.strategies);
    };
    fetch();
  }, []);

  const renderGeneralSettings = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-gray-900 dark:text-gray-100 mb-3 font-medium">Theme</label>
        <div className="grid grid-cols-3 gap-3">
          {(['light', 'dark', 'system'] as const).map((t) => {
            const icons = { light: SunIcon, dark: MoonIcon, system: ComputerDesktopIcon };
            const Icon = icons[t];
            return (
              <button
                key={t}
                onClick={() => setTheme(t)}
                className={`flex flex-col items-center p-4 rounded-lg border-2 transition-colors ${
                  theme === t
                    ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                    : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 text-gray-700 dark:text-gray-300'
                }`}
              >
                <Icon className="h-6 w-6 mb-2" />
                <span className="text-sm font-medium capitalize">{t}</span>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );

  const renderTradingConfig = () => {
    if (!tradingConfig) return <div className="text-gray-500">Loading...</div>;

    const items = [
      { label: 'Trading Mode', value: tradingConfig.is_paper_trading ? 'Paper Trading' : 'Live Trading', badge: tradingConfig.is_paper_trading ? 'blue' : 'red' },
      { label: 'Initial Capital', value: `₹${tradingConfig.initial_capital?.toLocaleString()}`, badge: null },
      { label: 'Brokerage', value: `${(tradingConfig.brokerage_charges * 100).toFixed(2)}%`, badge: null },
      { label: 'Auto Execute', value: tradingConfig.auto_execute ? 'Enabled' : 'Disabled', badge: tradingConfig.auto_execute ? 'green' : 'gray' },
      { label: 'Circuit Breaker', value: tradingConfig.circuit_breaker ? 'Active' : 'Inactive', badge: tradingConfig.circuit_breaker ? 'red' : 'green' },
      { label: 'Time Stop', value: `${tradingConfig.time_stop_days || 15} days`, badge: null },
      { label: 'Multiple Positions', value: tradingConfig.allow_multiple_positions_same_stock ? 'Allowed' : 'Not Allowed', badge: tradingConfig.allow_multiple_positions_same_stock ? 'green' : 'gray' },
    ];

    return (
      <div className="space-y-0">
        <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            These settings are read-only and managed via <code className="font-mono text-xs bg-yellow-100 dark:bg-yellow-800 px-1 rounded">backend/config.py</code>
          </p>
        </div>
        <div className="opacity-80">
          {items.map((item) => (
            <div key={item.label} className="flex items-center justify-between py-3 border-b border-gray-200 dark:border-gray-700 last:border-0 cursor-default select-none">
              <span className="text-gray-500 dark:text-gray-400 font-medium">{item.label}</span>
              {item.badge ? (
                <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                  item.badge === 'green' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' :
                  item.badge === 'red' ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300' :
                  item.badge === 'blue' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' :
                  'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                }`}>{item.value}</span>
              ) : (
                <span className="text-gray-500 dark:text-gray-400 font-semibold">{item.value}</span>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderStrategies = () => {
    if (strategies.length === 0) return <div className="text-gray-500">Loading...</div>;

    return (
      <>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Name</th>
                <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Status</th>
                <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Description</th>
                <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Risk/Trade</th>
                <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Max Pos</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
              {strategies.map((s) => (
                <tr
                  key={s.name}
                  onClick={() => setSelectedStrategy(s)}
                  className="hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors"
                >
                  <td className="px-4 py-3 font-semibold text-gray-900 dark:text-gray-100">{s.name}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                      s.enabled ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                    }`}>{s.enabled ? 'Enabled' : 'Disabled'}</span>
                  </td>
                  <td className="px-4 py-3 text-gray-600 dark:text-gray-400 max-w-xs truncate">{s.description || 'No description'}</td>
                  <td className="px-4 py-3 text-gray-700 dark:text-gray-300">{s.risk_management?.risk_per_trade_pct ?? '-'}%</td>
                  <td className="px-4 py-3 text-gray-700 dark:text-gray-300">{s.risk_management?.max_positions ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-gray-500 dark:text-gray-400">Click a strategy row to view full details</p>
      </>
    );
  };

  const renderContent = () => {
    switch (activeSection) {
      case 'general': return renderGeneralSettings();
      case 'trading': return renderTradingConfig();
      case 'strategies': return renderStrategies();
      default: return renderGeneralSettings();
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100 mb-4">Settings</h1>
        <p className="text-gray-600 dark:text-gray-300">Manage your preferences and view trading configuration</p>
      </div>

      <div className="grid lg:grid-cols-4 gap-8">
        <div className="lg:col-span-1">
          <nav className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
            <ul className="divide-y divide-gray-200 dark:divide-gray-700">
              {settingsSections.map((section) => {
                const Icon = section.icon;
                return (
                  <li key={section.id}>
                    <button
                      onClick={() => setActiveSection(section.id)}
                      className={`w-full flex items-center px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors ${
                        activeSection === section.id
                          ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border-r-2 border-blue-500'
                          : 'text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      <Icon className="h-5 w-5 mr-3" />
                      {section.name}
                    </button>
                  </li>
                );
              })}
            </ul>
          </nav>
        </div>

        <div className="lg:col-span-3">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 border border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-6">
              {settingsSections.find(s => s.id === activeSection)?.name}
            </h2>
            {renderContent()}
          </div>
        </div>
      </div>

      {/* Strategy Detail Modal */}
      {selectedStrategy && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700 w-full max-w-3xl max-h-[85vh] overflow-hidden mx-4">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">{selectedStrategy.name}</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">{selectedStrategy.file_name}</p>
              </div>
              <button
                onClick={() => setSelectedStrategy(null)}
                className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>

            {/* Body */}
            <div className="p-6 overflow-y-auto max-h-[70vh] space-y-6">
              {/* Status badges */}
              <div className="flex items-center gap-3 flex-wrap">
                <span className={`px-3 py-1.5 rounded-full text-sm font-semibold ${
                  selectedStrategy.enabled
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                    : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                }`}>
                  {selectedStrategy.enabled ? 'Enabled' : 'Disabled'}
                </span>
                {selectedStrategy.version && (
                  <span className="px-3 py-1.5 rounded-full text-sm bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300">
                    v{selectedStrategy.version}
                  </span>
                )}
              </div>

              {/* Description */}
              {selectedStrategy.description && (
                <div>
                  <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-1">Description</h4>
                  <p className="text-gray-900 dark:text-gray-100 text-sm">{selectedStrategy.description}</p>
                </div>
              )}

              {/* Key metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {selectedStrategy.risk_management && (
                  <>
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                      <p className="text-xs text-gray-500 dark:text-gray-400">Risk/Trade</p>
                      <p className="text-lg font-bold text-gray-900 dark:text-gray-100">{selectedStrategy.risk_management.risk_per_trade_pct}%</p>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                      <p className="text-xs text-gray-500 dark:text-gray-400">Max Positions</p>
                      <p className="text-lg font-bold text-gray-900 dark:text-gray-100">{selectedStrategy.risk_management.max_positions}</p>
                    </div>
                    <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
                      <p className="text-xs text-gray-500 dark:text-gray-400">Max Position %</p>
                      <p className="text-lg font-bold text-gray-900 dark:text-gray-100">{selectedStrategy.risk_management.max_position_pct}%</p>
                    </div>
                  </>
                )}
              </div>

              {/* All top-level sections rendered as formatted JSON blocks */}
              {Object.entries(selectedStrategy)
                .filter(([key]) => !['name', 'enabled', 'description', 'version', 'file_name'].includes(key))
                .map(([key, value]) => (
                  <div key={key}>
                    <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2 capitalize">
                      {key.replace(/_/g, ' ')}
                    </h4>
                    {typeof value === 'object' && value !== null ? (
                      <pre className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 text-xs text-gray-700 dark:text-gray-300 overflow-x-auto whitespace-pre-wrap">
                        {JSON.stringify(value, null, 2)}
                      </pre>
                    ) : (
                      <p className="text-gray-900 dark:text-gray-100">{String(value)}</p>
                    )}
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
