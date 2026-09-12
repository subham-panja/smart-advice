'use client';

import { useState } from 'react';
import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ShieldCheckIcon,
  SparklesIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  CalendarDaysIcon,
  ArrowTopRightOnSquareIcon,
  BanknotesIcon,
  CheckCircleIcon,
  XCircleIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';
import { BacktestYearlyBreakdownItem } from '@/lib/api';

interface YearlyBreakdownTableProps {
  breakdown: BacktestYearlyBreakdownItem[];
  title?: string;
  subtitle?: string;
  onFilterYear?: (year: string) => void;
  highlightedYear?: string;
}

export default function YearlyBreakdownTable({
  breakdown,
  title = 'Year-by-Year Performance Breakdown (10-Year Portfolio)',
  subtitle = 'Multi-year historical timeline showing compounded annual returns, market stress testing, and strategy regime behavior',
  onFilterYear,
  highlightedYear,
}: YearlyBreakdownTableProps) {
  const [expandedYears, setExpandedYears] = useState<Record<string, boolean>>({});

  const toggleExpand = (year: string) => {
    setExpandedYears((prev) => ({
      ...prev,
      [year]: !prev[year],
    }));
  };

  const toggleExpandAll = () => {
    const allExpanded = breakdown.every((item) => expandedYears[item.year]);
    if (allExpanded) {
      setExpandedYears({});
    } else {
      const next: Record<string, boolean> = {};
      breakdown.forEach((item) => {
        next[item.year] = true;
      });
      setExpandedYears(next);
    }
  };

  if (!breakdown || breakdown.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500 dark:text-gray-400 bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700">
        <CalendarDaysIcon className="w-10 h-10 mx-auto text-gray-400 mb-2" />
        <p className="font-semibold text-gray-700 dark:text-gray-300">No yearly breakdown data available</p>
        <p className="text-xs text-gray-500 mt-1">This simulation session does not contain multi-year snapshot records.</p>
      </div>
    );
  }

  // Calculate summary metrics
  const activeYears = breakdown.filter((item) => item.year !== '2016-2017' && item.year !== '2016');
  const positiveYears = breakdown.filter((item) => item.return_pct > 0);
  const bestYear = [...breakdown].sort((a, b) => b.return_pct - a.return_pct)[0];
  const worstYear = [...breakdown].sort((a, b) => a.return_pct - b.return_pct)[0];
  const maxAnnualDD = Math.min(...breakdown.map((item) => item.max_drawdown_pct));

  const initialCap = breakdown[0]?.start_portfolio_value ?? 0;
  const finalCap = breakdown[breakdown.length - 1]?.end_portfolio_value ?? 0;
  const totalReturn = initialCap > 0 ? ((finalCap - initialCap) / initialCap) * 100 : 0;

  const getBadgeStyle = (badge?: string) => {
    switch (badge) {
      case 'Warmup Baseline':
        return 'bg-slate-100 text-slate-700 dark:bg-slate-800/80 dark:text-slate-300 border-slate-300 dark:border-slate-700';
      case 'Crash Defense':
        return 'bg-amber-50 text-amber-700 dark:bg-amber-950/60 dark:text-amber-300 border-amber-200 dark:border-amber-800';
      case 'Liquidity Crisis Defense':
      case 'Liquidity Shock':
        return 'bg-purple-50 text-purple-700 dark:bg-purple-950/60 dark:text-purple-300 border-purple-200 dark:border-purple-800';
      case 'COVID Resilience':
        return 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800';
      case 'Cyclical Surge':
      case 'Cyclical Expansion':
        return 'bg-cyan-50 text-cyan-700 dark:bg-cyan-950/60 dark:text-cyan-300 border-cyan-200 dark:border-cyan-800';
      case 'Rate Hike Pullback':
        return 'bg-rose-50 text-rose-700 dark:bg-rose-950/60 dark:text-rose-300 border-rose-200 dark:border-rose-800';
      case 'Multi-Bagger Supercycle':
        return 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-sm shadow-indigo-500/30 font-semibold border-transparent';
      case 'Broad Expansion':
      case 'Secular Trend':
        return 'bg-blue-50 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border-blue-200 dark:border-blue-800';
      case 'Defensive Trailing':
        return 'bg-sky-50 text-sky-700 dark:bg-sky-950/60 dark:text-sky-300 border-sky-200 dark:border-sky-800';
      case 'Active Momentum':
      case 'Momentum Expansion':
        return 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-200 border-emerald-300 dark:border-emerald-700';
      default:
        return 'bg-indigo-50 text-indigo-700 dark:bg-indigo-950/50 dark:text-indigo-300 border-indigo-200 dark:border-indigo-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner & Summary Cards */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-5 sm:p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 pb-5 border-b border-gray-100 dark:border-gray-700">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400">
                <CalendarDaysIcon className="w-5 h-5" />
              </div>
              <h2 className="text-lg sm:text-xl font-bold tracking-tight text-gray-900 dark:text-white">
                {title}
              </h2>
            </div>
            <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400 mt-1">
              {subtitle}
            </p>
          </div>

          <div className="flex items-center gap-2.5 self-start md:self-center">
            <button
              onClick={toggleExpandAll}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 text-xs font-semibold text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors shadow-sm"
            >
              {breakdown.every((item) => expandedYears[item.year]) ? (
                <>
                  <ChevronUpIcon className="w-3.5 h-3.5" />
                  Collapse All
                </>
              ) : (
                <>
                  <ChevronDownIcon className="w-3.5 h-3.5" />
                  Expand All Details
                </>
              )}
            </button>
          </div>
        </div>

        {/* Quick Highlights Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 mt-5">
          <div className="bg-gray-50 dark:bg-gray-900/70 rounded-xl p-3 sm:p-3.5 border border-gray-100 dark:border-gray-700">
            <span className="text-[11px] font-medium text-gray-500 dark:text-gray-400 block">Overall Compounding</span>
            <div className="flex items-baseline gap-1.5 mt-0.5">
              <span className="text-lg sm:text-xl font-bold text-emerald-600 dark:text-emerald-400">
                +{totalReturn.toFixed(1)}%
              </span>
              <span className="text-[11px] text-gray-400">
                (₹{initialCap.toLocaleString('en-IN', { maximumFractionDigits: 0 })} → ₹{finalCap.toLocaleString('en-IN', { maximumFractionDigits: 0 })})
              </span>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900/70 rounded-xl p-3 sm:p-3.5 border border-gray-100 dark:border-gray-700">
            <span className="text-[11px] font-medium text-gray-500 dark:text-gray-400 block">Best Supercycle Year</span>
            <div className="flex items-baseline gap-1.5 mt-0.5">
              <span className="text-lg sm:text-xl font-bold text-indigo-600 dark:text-indigo-400">
                +{bestYear?.return_pct?.toFixed(1)}%
              </span>
              <span className="text-xs font-semibold text-gray-700 dark:text-gray-300">
                ({bestYear?.display_year})
              </span>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900/70 rounded-xl p-3 sm:p-3.5 border border-gray-100 dark:border-gray-700">
            <span className="text-[11px] font-medium text-gray-500 dark:text-gray-400 block">Max Annual Drawdown</span>
            <div className="flex items-baseline gap-1.5 mt-0.5">
              <span className="text-lg sm:text-xl font-bold text-amber-600 dark:text-amber-400">
                {maxAnnualDD.toFixed(1)}%
              </span>
              <span className="text-[11px] text-gray-400">(COVID 2020 Peak)</span>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900/70 rounded-xl p-3 sm:p-3.5 border border-gray-100 dark:border-gray-700">
            <span className="text-[11px] font-medium text-gray-500 dark:text-gray-400 block">Positive / Defended Years</span>
            <div className="flex items-baseline gap-1.5 mt-0.5">
              <span className="text-lg sm:text-xl font-bold text-gray-900 dark:text-white">
                {positiveYears.length} / {breakdown.length}
              </span>
              <span className="text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold">
                ({((positiveYears.length / (breakdown.length || 1)) * 100).toFixed(0)}% Win)
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Interactive Table */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm border-collapse">
            <thead className="bg-gray-50/95 dark:bg-gray-900/95 backdrop-blur text-gray-600 dark:text-gray-300 uppercase text-[11px] tracking-wider border-b border-gray-200 dark:border-gray-700 font-semibold">
              <tr>
                <th className="py-3.5 px-4 sm:px-6 w-[140px]">Year</th>
                <th className="py-3.5 px-4 text-right w-[120px]">Return</th>
                <th className="py-3.5 px-4 text-right hidden md:table-cell w-[170px]">Portfolio Growth</th>
                <th className="py-3.5 px-4 text-right hidden sm:table-cell w-[100px]">Max DD</th>
                <th className="py-3.5 px-4 text-right hidden lg:table-cell w-[120px]">Win Rate / Trades</th>
                <th className="py-3.5 px-4 sm:px-6">Market Context & Strategy Behavior</th>
                <th className="py-3.5 px-3 text-center w-[60px]">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-700/80">
              {breakdown.map((row) => {
                const isExpanded = !!expandedYears[row.year];
                const isPositive = row.return_pct > 0;
                const isNeutral = Math.abs(row.return_pct) < 0.001;
                const isHighlighted = highlightedYear && row.year.includes(highlightedYear);

                return (
                  <tr
                    key={row.year}
                    className={`transition-colors group ${
                      isHighlighted
                        ? 'bg-indigo-50/60 dark:bg-indigo-950/40'
                        : isExpanded
                        ? 'bg-gray-50/70 dark:bg-gray-900/70'
                        : 'hover:bg-gray-50/50 dark:hover:bg-gray-700/40'
                    }`}
                  >
                    {/* Year Column */}
                    <td className="py-4 px-4 sm:px-6 align-top">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-gray-900 dark:text-white tracking-tight text-sm sm:text-base">
                          {row.display_year}
                        </span>
                        {row.year === '2023' && (
                          <span className="p-0.5 rounded text-amber-500" title="Supercycle Winner">
                            <SparklesIcon className="w-4 h-4" />
                          </span>
                        )}
                      </div>
                      <div className="text-[11px] text-gray-400 mt-0.5">
                        {row.trading_days} sessions
                      </div>
                    </td>

                    {/* Return Column */}
                    <td className="py-4 px-4 text-right align-top">
                      <div
                        className={`inline-flex items-center gap-1 font-bold text-sm sm:text-base ${
                          isPositive
                            ? 'text-emerald-600 dark:text-emerald-400'
                            : isNeutral
                            ? 'text-gray-500 dark:text-gray-400'
                            : 'text-rose-600 dark:text-rose-400'
                        }`}
                      >
                        {isPositive ? (
                          <ArrowTrendingUpIcon className="w-4 h-4 shrink-0" />
                        ) : isNeutral ? (
                          <span className="text-gray-400">•</span>
                        ) : (
                          <ArrowTrendingDownIcon className="w-4 h-4 shrink-0" />
                        )}
                        <span>
                          {isPositive ? '+' : ''}
                          {row.return_pct.toFixed(2)}%
                        </span>
                      </div>
                      <div className="text-[10px] text-gray-400 mt-0.5">
                        {isPositive && row.realized_pnl > 0 ? `+₹${row.realized_pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })} net` : ' '}
                      </div>
                    </td>

                    {/* Capital Growth Column */}
                    <td className="py-4 px-4 text-right align-top hidden md:table-cell">
                      <div className="text-xs font-semibold text-gray-800 dark:text-gray-200">
                        ₹{row.end_portfolio_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                      </div>
                      <div className="text-[11px] text-gray-400 mt-0.5">
                        from ₹{row.start_portfolio_value.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                      </div>
                    </td>

                    {/* Max Drawdown Column */}
                    <td className="py-4 px-4 text-right align-top hidden sm:table-cell">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${
                          row.max_drawdown_pct < -20
                            ? 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300'
                            : row.max_drawdown_pct < -10
                            ? 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300'
                            : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                        }`}
                      >
                        {row.max_drawdown_pct.toFixed(1)}%
                      </span>
                    </td>

                    {/* Win Rate / Trades Column */}
                    <td className="py-4 px-4 text-right align-top hidden lg:table-cell">
                      <div className="text-xs font-semibold text-gray-800 dark:text-gray-200">
                        {row.exits_count > 0 ? `${row.win_rate.toFixed(1)}%` : '—'}
                      </div>
                      <div className="text-[11px] text-gray-400 mt-0.5">
                        {row.exits_count} {row.exits_count === 1 ? 'exit' : 'exits'} ({row.total_events} events)
                      </div>
                    </td>

                    {/* Market Context & Strategy Behavior Column */}
                    <td className="py-4 px-4 sm:px-6 align-top">
                      <div className="flex flex-wrap items-center gap-2 mb-1.5">
                        {row.market_badge && (
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold border ${getBadgeStyle(
                              row.market_badge
                            )}`}
                          >
                            {row.market_badge}
                          </span>
                        )}
                        {row.regime && (
                          <span className="text-[11px] text-gray-400 dark:text-gray-500 font-medium">
                            • {row.regime}
                          </span>
                        )}
                      </div>
                      <p className="text-xs sm:text-sm font-medium text-gray-900 dark:text-gray-100 leading-relaxed">
                        {row.market_context}
                      </p>

                      {/* Expandable Details Tray */}
                      {isExpanded && (
                        <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 space-y-4">
                          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                            {/* Best Trade Card */}
                            {row.best_trade ? (
                              <div className="p-3 rounded-xl bg-emerald-50/80 dark:bg-emerald-950/30 border border-emerald-200/80 dark:border-emerald-900/60">
                                <div className="flex items-center justify-between text-xs text-emerald-800 dark:text-emerald-300 font-semibold mb-1">
                                  <span className="flex items-center gap-1">
                                    <CheckCircleIcon className="w-3.5 h-3.5 text-emerald-600" />
                                    Top Winning Trade
                                  </span>
                                  <span>+{row.best_trade.pnl_pct.toFixed(1)}%</span>
                                </div>
                                <div className="text-base font-bold text-gray-900 dark:text-white">
                                  {row.best_trade.symbol}
                                </div>
                                <div className="flex items-center justify-between text-[11px] text-emerald-700 dark:text-emerald-300 mt-1">
                                  <span>+₹{row.best_trade.pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
                                  <span className="text-gray-500 dark:text-gray-400">
                                    {row.best_trade.exit_reason || row.best_trade.exit_date}
                                  </span>
                                </div>
                              </div>
                            ) : (
                              <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/70 border border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400 flex items-center justify-center">
                                No closed trades in this period
                              </div>
                            )}

                            {/* Worst Trade Card */}
                            {row.worst_trade ? (
                              <div className="p-3 rounded-xl bg-rose-50/80 dark:bg-rose-950/30 border border-rose-200/80 dark:border-rose-900/60">
                                <div className="flex items-center justify-between text-xs text-rose-800 dark:text-rose-300 font-semibold mb-1">
                                  <span className="flex items-center gap-1">
                                    <XCircleIcon className="w-3.5 h-3.5 text-rose-600" />
                                    Largest Loss Defended
                                  </span>
                                  <span>{row.worst_trade.pnl_pct.toFixed(1)}%</span>
                                </div>
                                <div className="text-base font-bold text-gray-900 dark:text-white">
                                  {row.worst_trade.symbol}
                                </div>
                                <div className="flex items-center justify-between text-[11px] text-rose-700 dark:text-rose-300 mt-1">
                                  <span>₹{row.worst_trade.pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}</span>
                                  <span className="text-gray-500 dark:text-gray-400">
                                    {row.worst_trade.exit_reason || row.worst_trade.exit_date}
                                  </span>
                                </div>
                              </div>
                            ) : (
                              <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/70 border border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400 flex items-center justify-center">
                                Zero losing exits
                              </div>
                            )}

                            {/* Multi-Baggers & Capital Profile */}
                            <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/70 border border-gray-200 dark:border-gray-700">
                              <span className="text-xs font-semibold text-gray-700 dark:text-gray-300 block mb-1">
                                Equity Range in {row.display_year}
                              </span>
                              <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                                <div className="flex justify-between">
                                  <span>Peak Value:</span>
                                  <span className="font-semibold text-gray-900 dark:text-white">
                                    ₹{Number(row.peak_portfolio_value || row.end_portfolio_value).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span>Annual Realized Gain:</span>
                                  <span className={`font-semibold ${row.realized_pnl >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                                    {row.realized_pnl >= 0 ? '+' : ''}₹{row.realized_pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* Top Gainers Tickers List */}
                          {row.top_gainers && row.top_gainers.length > 0 && (
                            <div className="flex flex-wrap items-center gap-2 pt-1">
                              <span className="text-xs font-semibold text-gray-500 dark:text-gray-400">
                                Key Momentum Contributors:
                              </span>
                              {row.top_gainers.map((g) => (
                                <span
                                  key={g.symbol}
                                  className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 text-xs font-semibold border border-indigo-200 dark:border-indigo-800"
                                >
                                  <span>{g.symbol}</span>
                                  <span className="text-[10px] text-emerald-600 dark:text-emerald-400 font-bold">
                                    +₹{g.pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                                  </span>
                                </span>
                              ))}
                            </div>
                          )}

                          {/* Action Button: Filter Journal by Year */}
                          {onFilterYear && row.exits_count > 0 && (
                            <div className="pt-2">
                              <button
                                onClick={() => onFilterYear(row.year === '2016-2017' ? '2017' : row.year)}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-semibold text-xs transition-colors shadow-sm"
                              >
                                <ArrowTopRightOnSquareIcon className="w-3.5 h-3.5" />
                                Inspect {row.display_year} Trades in Journal ({row.exits_count})
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </td>

                    {/* Expand/Collapse Action Icon */}
                    <td className="py-4 px-3 text-center align-top">
                      <button
                        onClick={() => toggleExpand(row.year)}
                        className="p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                        title={isExpanded ? 'Collapse' : 'Expand Details'}
                      >
                        {isExpanded ? (
                          <ChevronUpIcon className="w-4 h-4" />
                        ) : (
                          <ChevronDownIcon className="w-4 h-4" />
                        )}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
