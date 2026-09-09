'use client';

import { useState, useEffect, useMemo } from 'react';
import Link from 'next/link';
import {
  ChartBarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ShieldCheckIcon,
  ClockIcon,
  ArrowPathIcon,
  TrashIcon,
  DocumentMagnifyingGlassIcon,
  ChevronRightIcon,
  CircleStackIcon,
  SparklesIcon,
  ChevronUpDownIcon,
  ChevronUpIcon,
  ChevronDownIcon,
} from '@heroicons/react/24/outline';
import { getBacktestSessions, deleteBacktestSession, BacktestSession } from '@/lib/api';

export default function BacktestsPage() {
  const [sessions, setSessions] = useState<BacktestSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const [sessSortBy, setSessSortBy] = useState<
    'strategy_name' | 'total_return_pct' | 'cagr' | 'max_drawdown_pct' | 'sharpe_ratio' | 'win_rate' | 'total_trades' | 'final_portfolio_value'
  >('total_return_pct');
  const [sessSortDir, setSessSortDir] = useState<'desc' | 'asc'>('desc');

  const handleSessSort = (field: typeof sessSortBy) => {
    if (sessSortBy === field) {
      setSessSortDir((prev) => (prev === 'desc' ? 'asc' : 'desc'));
    } else {
      setSessSortBy(field);
      setSessSortDir(field === 'strategy_name' ? 'asc' : 'desc');
    }
  };

  const renderSessSortIcon = (field: typeof sessSortBy) => {
    if (sessSortBy === field) {
      return sessSortDir === 'asc' ? (
        <ChevronUpIcon className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400 shrink-0" />
      ) : (
        <ChevronDownIcon className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400 shrink-0" />
      );
    }
    return (
      <ChevronUpDownIcon className="w-3.5 h-3.5 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
    );
  };

  const sortedSessions = useMemo(() => {
    return [...sessions].sort((a, b) => {
      const valA = a[sessSortBy] ?? 0;
      const valB = b[sessSortBy] ?? 0;
      if (typeof valA === 'string' && typeof valB === 'string') {
        return sessSortDir === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return sessSortDir === 'asc' ? Number(valA) - Number(valB) : Number(valB) - Number(valA);
    });
  }, [sessions, sessSortBy, sessSortDir]);


  const fetchSessions = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getBacktestSessions();
      if (res.status === 'success') {
        setSessions(res.sessions || []);
      } else {
        setError(res.error || 'Failed to load backtest sessions');
      }
    } catch (err: any) {
      setError(err?.message || 'Error connecting to API');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!confirm('Are you sure you want to delete this backtest session and all its recorded trades?')) {
      return;
    }
    setDeletingId(id);
    try {
      const res = await deleteBacktestSession(id);
      if (res.status === 'success') {
        setSessions((prev) => prev.filter((s) => s._id !== id));
      } else {
        alert(res.error || 'Failed to delete backtest session');
      }
    } catch (err: any) {
      alert(err?.message || 'Error deleting session');
    } finally {
      setDeletingId(null);
    }
  };

  // Top/Latest session
  const latestSession = sessions.length > 0 ? sessions[0] : null;
  const metrics = latestSession?.summary_metrics || {};

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-gray-200 dark:border-gray-800 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-indigo-500 to-purple-600 text-white shadow-md shadow-indigo-500/20">
              <ChartBarIcon className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
                Portfolio Backtests
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                Explore multi-year historical simulation sessions and inspect execution trade journals
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={fetchSessions}
            disabled={loading}
            className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors shadow-sm disabled:opacity-50"
          >
            <ArrowPathIcon className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="p-4 rounded-xl bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900/60 text-red-700 dark:text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Latest Run Spotlight Banner */}
      {latestSession && (
        <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-indigo-950 via-gray-900 to-slate-900 text-white p-6 sm:p-8 shadow-xl border border-indigo-900/50">
          <div className="absolute top-0 right-0 -mt-8 -mr-8 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6 relative z-10">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 mb-3">
                <SparklesIcon className="w-3.5 h-3.5" />
                Latest Simulation Spotlight
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">
                {latestSession.strategy_name}
              </h2>
              <div className="flex flex-wrap items-center gap-4 text-xs sm:text-sm text-gray-300 mt-2">
                <span className="flex items-center gap-1.5">
                  <ClockIcon className="w-4 h-4 text-gray-400" />
                  {latestSession.date_range?.start_date || 'N/A'} → {latestSession.date_range?.end_date || 'N/A'}
                </span>
                <span>•</span>
                <span className="flex items-center gap-1.5">
                  <CircleStackIcon className="w-4 h-4 text-gray-400" />
                  {latestSession.total_symbols || 0} Stocks Universe
                </span>
                <span>•</span>
                <span className="capitalize px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-medium">
                  {latestSession.status}
                </span>
              </div>
            </div>

            <Link
              href={`/backtests/${latestSession._id}`}
              className="inline-flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-indigo-500 hover:bg-indigo-600 text-white font-semibold text-sm transition-all shadow-lg shadow-indigo-500/30 hover:shadow-indigo-500/50 self-start lg:self-center"
            >
              <DocumentMagnifyingGlassIcon className="w-5 h-5" />
              Inspect Full Trade Journal
              <ChevronRightIcon className="w-4 h-4" />
            </Link>
          </div>

          {/* Quick Metrics Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 sm:gap-4 mt-6 pt-6 border-t border-gray-800/80">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-3 sm:p-4 border border-white/5">
              <span className="text-xs text-gray-400 block">Total Return</span>
              <span
                className={`text-lg sm:text-xl font-bold ${
                  (latestSession.total_return_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
                }`}
              >
                {(latestSession.total_return_pct ?? 0) >= 0 ? '+' : ''}
                {Number(latestSession.total_return_pct ?? 0).toFixed(1)}%
              </span>
            </div>

            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-3 sm:p-4 border border-white/5">
              <span className="text-xs text-gray-400 block">CAGR</span>
              <span className="text-lg sm:text-xl font-bold text-white">
                {Number(latestSession.cagr ?? 0).toFixed(2)}%
              </span>
            </div>

            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-3 sm:p-4 border border-white/5">
              <span className="text-xs text-gray-400 block">Sharpe Ratio</span>
              <span className="text-lg sm:text-xl font-bold text-white">
                {Number(latestSession.sharpe_ratio ?? 0).toFixed(2)}
              </span>
            </div>

            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-3 sm:p-4 border border-white/5">
              <span className="text-xs text-gray-400 block">Max Drawdown</span>
              <span className="text-lg sm:text-xl font-bold text-amber-400">
                {Number(latestSession.max_drawdown_pct ?? 0).toFixed(1)}%
              </span>
            </div>

            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-3 sm:p-4 border border-white/5">
              <span className="text-xs text-gray-400 block">Win Rate</span>
              <span className="text-lg sm:text-xl font-bold text-white">
                {Number(latestSession.win_rate ?? 0).toFixed(1)}%
              </span>
            </div>

            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-3 sm:p-4 border border-white/5">
              <span className="text-xs text-gray-400 block">Profit Factor</span>
              <span className="text-lg sm:text-xl font-bold text-emerald-400">
                {Number(latestSession.profit_factor ?? 0).toFixed(2)}x
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Sessions Table Section */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="p-5 sm:p-6 border-b border-gray-200 dark:border-gray-700 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <h2 className="text-lg font-bold text-gray-900 dark:text-white">
              Simulation Sessions
            </h2>
            <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400">
              {sessions.length} backtest {sessions.length === 1 ? 'session' : 'sessions'} recorded
            </p>
          </div>
        </div>

        {loading ? (
          <div className="p-12 text-center text-gray-500 dark:text-gray-400">
            <ArrowPathIcon className="w-8 h-8 animate-spin mx-auto mb-3 text-indigo-500" />
            <p>Loading backtest sessions...</p>
          </div>
        ) : sessions.length === 0 ? (
          <div className="p-12 text-center text-gray-500 dark:text-gray-400 space-y-3">
            <DocumentMagnifyingGlassIcon className="w-12 h-12 mx-auto text-gray-300 dark:text-gray-600" />
            <p className="text-base font-semibold text-gray-700 dark:text-gray-300">No Backtest Sessions Found</p>
            <p className="text-sm max-w-md mx-auto">
              Run a portfolio backtest from the CLI with <code className="px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-xs">python scripts/run_portfolio_backtest.py --strategy Swing_Trading_v2 --period 5y</code>.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-50 dark:bg-gray-850 text-gray-500 dark:text-gray-400 uppercase text-xs tracking-wider border-b border-gray-200 dark:border-gray-700 font-semibold">
                <tr>
                  <th
                    onClick={() => handleSessSort('strategy_name')}
                    className="py-3.5 px-4 sm:px-6 cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Strategy"
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Strategy & Range</span>
                      {renderSessSortIcon('strategy_name')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSessSort('total_return_pct')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Total Return"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Return %</span>
                      {renderSessSortIcon('total_return_pct')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSessSort('cagr')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by CAGR"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>CAGR</span>
                      {renderSessSortIcon('cagr')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSessSort('max_drawdown_pct')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Max Drawdown"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Max DD</span>
                      {renderSessSortIcon('max_drawdown_pct')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSessSort('sharpe_ratio')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Sharpe Ratio"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Sharpe</span>
                      {renderSessSortIcon('sharpe_ratio')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSessSort('win_rate')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Win Rate"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Win Rate</span>
                      {renderSessSortIcon('win_rate')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSessSort('total_trades')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Total Trades"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Trades</span>
                      {renderSessSortIcon('total_trades')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSessSort('final_portfolio_value')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Final Portfolio Value"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Final Value</span>
                      {renderSessSortIcon('final_portfolio_value')}
                    </div>
                  </th>
                  <th className="py-3.5 px-4 sm:px-6 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-750">
                {sortedSessions.map((s) => {
                  const returnPct = s.total_return_pct ?? 0;
                  const isPositive = returnPct >= 0;

                  return (
                    <tr
                      key={s._id}
                      className="hover:bg-gray-50/80 dark:hover:bg-gray-750/50 transition-colors group cursor-pointer"
                    >
                      <td className="py-4 px-4 sm:px-6">
                        <Link href={`/backtests/${s._id}`} className="block">
                          <div className="font-semibold text-gray-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                            {s.strategy_name}
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 flex items-center gap-1.5">
                            <ClockIcon className="w-3.5 h-3.5" />
                            {s.date_range?.start_date || 'N/A'} → {s.date_range?.end_date || 'N/A'}
                          </div>
                        </Link>
                      </td>

                      <td className="py-4 px-4 text-right font-semibold">
                        <span
                          className={`inline-flex items-center gap-0.5 ${
                            isPositive ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
                          }`}
                        >
                          {isPositive ? (
                            <ArrowTrendingUpIcon className="w-3.5 h-3.5" />
                          ) : (
                            <ArrowTrendingDownIcon className="w-3.5 h-3.5" />
                          )}
                          {isPositive ? '+' : ''}
                          {Number(returnPct).toFixed(1)}%
                        </span>
                      </td>

                      <td className="py-4 px-4 text-right font-medium text-gray-900 dark:text-gray-100">
                        {s.cagr !== undefined ? `${Number(s.cagr).toFixed(1)}%` : '—'}
                      </td>

                      <td className="py-4 px-4 text-right font-medium text-amber-600 dark:text-amber-400">
                        {s.max_drawdown_pct !== undefined ? `${Number(s.max_drawdown_pct).toFixed(1)}%` : '—'}
                      </td>

                      <td className="py-4 px-4 text-right font-medium text-gray-900 dark:text-gray-100">
                        {s.sharpe_ratio !== undefined ? Number(s.sharpe_ratio).toFixed(2) : '—'}
                      </td>

                      <td className="py-4 px-4 text-right font-medium text-gray-900 dark:text-gray-100">
                        {s.win_rate !== undefined ? `${Number(s.win_rate).toFixed(1)}%` : '—'}
                      </td>

                      <td className="py-4 px-4 text-right font-medium text-gray-700 dark:text-gray-300">
                        {s.total_trades ?? '—'}
                      </td>

                      <td className="py-4 px-4 text-right font-semibold text-gray-900 dark:text-gray-100">
                        {s.final_portfolio_value !== undefined
                          ? `₹${Number(s.final_portfolio_value).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
                          : '—'}
                      </td>

                      <td className="py-4 px-4 sm:px-6 text-center">
                        <div className="inline-flex items-center gap-2">
                          <Link
                            href={`/backtests/${s._id}`}
                            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 hover:bg-indigo-100 dark:hover:bg-indigo-900/60 font-medium text-xs transition-colors"
                          >
                            <DocumentMagnifyingGlassIcon className="w-3.5 h-3.5" />
                            Journal
                          </Link>

                          <button
                            onClick={(e) => handleDelete(s._id, e)}
                            disabled={deletingId === s._id}
                            title="Delete Session"
                            className="p-1.5 rounded-lg text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/40 transition-colors"
                          >
                            <TrashIcon className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
