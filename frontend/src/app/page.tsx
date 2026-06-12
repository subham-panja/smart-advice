'use client';

import { useState, useEffect } from 'react';
import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  BanknotesIcon,
  ChartPieIcon,
  FireIcon,
  PlusCircleIcon,
  ShieldCheckIcon,
  FlagIcon,
  XCircleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import { getDashboardStats, DashboardStats } from '@/lib/api';

export default function Home() {
  const [stats, setStats] = useState<Partial<DashboardStats>>({});
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    const res = await getDashboardStats();
    if (res.status === 'success') setStats(res);
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  const portfolio = stats.portfolio || {
    total_equity: 0,
    total_invested: 0,
    cash_remaining: 0,
    deployed_pct: 0,
    initial_capital: 100000,
    total_pnl: 0,
    pnl_pct: 0,
    realized_pnl: 0,
    unrealized_pnl: 0,
    open_positions: 0,
    total_trades: 0,
  };

  const performance = stats.performance || {
    win_rate: 0,
    profit_factor: 0,
    avg_win_pct: 0,
    avg_loss_pct: 0,
    wins: 0,
    losses: 0,
    total_closed: 0,
  };

  const today = stats.today || {
    trades_opened: 0,
    pyramids_added: 0,
    sl_trails: 0,
    targets_hit: 0,
    positions_closed: 0,
  };

  const positions = stats.positions || [];
  const activityFeed = stats.activity_feed || [];

  const formatCurrency = (val: number) => {
    if (Math.abs(val) >= 100000) {
      return `₹${(val / 100000).toFixed(2)}L`;
    }
    return `₹${val.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  };

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'POSITION_OPENED':
        return <PlusCircleIcon className="h-4 w-4 text-green-600" />;
      case 'PYRAMID':
        return <ArrowTrendingUpIcon className="h-4 w-4 text-blue-600" />;
      case 'TRAIL_SL':
        return <ShieldCheckIcon className="h-4 w-4 text-amber-600" />;
      case 'TARGET_HIT':
        return <FlagIcon className="h-4 w-4 text-purple-600" />;
      case 'CLOSED':
        return <XCircleIcon className="h-4 w-4 text-red-600" />;
      default:
        return <FireIcon className="h-4 w-4 text-gray-600" />;
    }
  };

  const formatTimestamp = (ts: string) => {
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Dashboard</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Portfolio overview and trading performance
          </p>
        </div>
        <button
          onClick={fetchData}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition"
        >
          <ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Portfolio KPIs */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          label="Total Equity"
          value={formatCurrency(portfolio.total_equity)}
          subtitle={`${portfolio.pnl_pct >= 0 ? '+' : ''}${portfolio.pnl_pct.toFixed(2)}%`}
          positive={portfolio.pnl_pct >= 0}
          icon={portfolio.pnl_pct >= 0 ? ArrowTrendingUpIcon : ArrowTrendingDownIcon}
        />
        <KPICard
          label="Total P&L"
          value={formatCurrency(portfolio.total_pnl)}
          subtitle={`Realized: ${formatCurrency(portfolio.realized_pnl)}`}
          positive={portfolio.total_pnl >= 0}
          icon={BanknotesIcon}
        />
        <KPICard
          label="Cash Available"
          value={formatCurrency(portfolio.cash_remaining)}
          subtitle={`${portfolio.deployed_pct.toFixed(1)}% deployed`}
          icon={ChartPieIcon}
        />
        <KPICard
          label="Open Positions"
          value={String(portfolio.open_positions)}
          subtitle={`${portfolio.total_trades} total trades`}
          icon={FireIcon}
        />
      </div>

      {/* Performance & Today's Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Metrics */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Performance Metrics
          </h2>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600 dark:text-gray-400">Win Rate</span>
              <div className="flex items-center gap-3">
                <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-500 transition-all"
                    style={{ width: `${performance.win_rate}%` }}
                  />
                </div>
                <span className="text-sm font-semibold text-gray-900 dark:text-gray-100 w-12 text-right">
                  {performance.win_rate.toFixed(1)}%
                </span>
              </div>
            </div>
            <MetricRow label="Profit Factor" value={performance.profit_factor.toFixed(2)} />
            <MetricRow label="Avg Win" value={`${performance.avg_win_pct.toFixed(2)}%`} positive />
            <MetricRow label="Avg Loss" value={`${performance.avg_loss_pct.toFixed(2)}%`} negative />
            <MetricRow label="Wins / Losses" value={`${performance.wins} / ${performance.losses}`} />
            <MetricRow label="Total Closed" value={String(performance.total_closed)} />
          </div>
        </div>

        {/* Today's Activity */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Today's Activity
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <ActivityCard label="Trades Opened" value={today.trades_opened} icon={PlusCircleIcon} color="green" />
            <ActivityCard label="Pyramids Added" value={today.pyramids_added} icon={ArrowTrendingUpIcon} color="blue" />
            <ActivityCard label="SL Trails" value={today.sl_trails} icon={ShieldCheckIcon} color="amber" />
            <ActivityCard label="Targets Hit" value={today.targets_hit} icon={FlagIcon} color="purple" />
            <ActivityCard label="Positions Closed" value={today.positions_closed} icon={XCircleIcon} color="red" />
          </div>
        </div>
      </div>

      {/* Open Positions */}
      {positions.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-700 bg-gradient-to-r from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
            <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100">
              Open Positions
              <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
                ({positions.length})
              </span>
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50/50 dark:bg-gray-900/50">
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Symbol</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Qty</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Entry</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Current</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">P&L</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">Investment</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700/50">
                {positions.map((pos) => (
                  <tr key={pos.symbol} className="hover:bg-gray-50/50 dark:hover:bg-gray-700/30 transition-colors">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-gray-900 dark:text-gray-100">{pos.symbol}</span>
                        {pos.adds_count > 0 && (
                          <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded">
                            +{pos.adds_count}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right font-medium text-gray-900 dark:text-gray-100">{pos.quantity}</td>
                    <td className="px-4 py-3 text-right text-gray-600 dark:text-gray-400">₹{pos.entry_price.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right text-gray-600 dark:text-gray-400">₹{pos.current_price.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right">
                      <div className={`font-semibold ${pos.pnl_pct >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {pos.pnl_pct >= 0 ? '+' : ''}{pos.pnl_pct.toFixed(2)}%
                      </div>
                      <div className={`text-xs ${pos.unrealized_pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {pos.unrealized_pnl >= 0 ? '+' : ''}₹{pos.unrealized_pnl.toLocaleString('en-IN')}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right text-gray-600 dark:text-gray-400">
                      ₹{pos.total_investment.toLocaleString('en-IN')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Activity Feed */}
      {activityFeed.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Recent Activity
          </h2>
          <div className="space-y-3">
            {activityFeed.slice(0, 10).map((log, idx) => (
              <div key={idx} className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
                {getActionIcon(log.action)}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-gray-900 dark:text-gray-100">{log.symbol}</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">{log.action}</span>
                  </div>
                </div>
                <span className="text-xs text-gray-400 dark:text-gray-500 whitespace-nowrap">
                  {formatTimestamp(log.timestamp)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function KPICard({
  label,
  value,
  subtitle,
  positive,
  icon: Icon,
}: {
  label: string;
  value: string;
  subtitle?: string;
  positive?: boolean;
  icon?: any;
}) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
      <div className="flex items-start justify-between">
        <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
          {label}
        </p>
        {Icon && (
          <Icon
            className={`h-5 w-5 ${
              positive !== undefined
                ? positive
                  ? 'text-green-500'
                  : 'text-red-500'
                : 'text-gray-400 dark:text-gray-500'
            }`}
          />
        )}
      </div>
      <p
        className={`text-2xl font-bold mt-2 ${
          positive !== undefined
            ? positive
              ? 'text-green-600 dark:text-green-400'
              : 'text-red-600 dark:text-red-400'
            : 'text-gray-900 dark:text-gray-100'
        }`}
      >
        {value}
      </p>
      {subtitle && (
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{subtitle}</p>
      )}
    </div>
  );
}

function MetricRow({ label, value, positive, negative }: { label: string; value: string; positive?: boolean; negative?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-gray-600 dark:text-gray-400">{label}</span>
      <span
        className={`text-sm font-semibold ${
          positive
            ? 'text-green-600 dark:text-green-400'
            : negative
            ? 'text-red-600 dark:text-red-400'
            : 'text-gray-900 dark:text-gray-100'
        }`}
      >
        {value}
      </span>
    </div>
  );
}

function ActivityCard({ label, value, icon: Icon, color }: { label: string; value: number; icon: any; color: string }) {
  const colorClasses = {
    green: 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400',
    blue: 'bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400',
    amber: 'bg-amber-100 dark:bg-amber-900/30 text-amber-600 dark:text-amber-400',
    purple: 'bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400',
    red: 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400',
  };

  return (
    <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
      <div className={`p-2 rounded-lg ${colorClasses[color as keyof typeof colorClasses]}`}>
        <Icon className="h-5 w-5" />
      </div>
      <div>
        <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
        <p className="text-lg font-bold text-gray-900 dark:text-gray-100">{value}</p>
      </div>
    </div>
  );
}
