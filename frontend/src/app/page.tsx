'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
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
  SparklesIcon,
  ArrowTopRightOnSquareIcon,
  EyeIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';
import {
  getDashboardStats,
  DashboardStats,
  getRecommendations,
  StockRecommendation,
  updatePosition,
  updateRecommendation,
} from '@/lib/api';
import StockPickNoteTag from './components/StockPickNoteTag';
import MotivationalQuotes from './components/MotivationalQuotes';
import DailyEarningsChart from './components/DailyEarningsChart';
import StockAnalysisDeepDive from './components/StockAnalysisDeepDive';
import { cleanSymbol, formatDecimal } from '@/lib/utils';

export default function Home() {
  const [stats, setStats] = useState<Partial<DashboardStats>>({});
  const [recommendations, setRecommendations] = useState<StockRecommendation[]>([]);
  const [selectedRecommendation, setSelectedRecommendation] = useState<StockRecommendation | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [statsRes, recsRes] = await Promise.all([
        getDashboardStats().catch(() => ({ status: 'error' } as any)),
        getRecommendations().catch(() => ({ status: 'error', recommendations: [] })),
      ]);

      if (statsRes && statsRes.status === 'success') {
        setStats(statsRes);
      }

      if (recsRes && recsRes.status === 'success' && recsRes.recommendations) {
        const recsList: StockRecommendation[] = recsRes.recommendations;
        setRecommendations(recsList.slice(0, 10));

        setSelectedRecommendation((prev) => {
          if (prev && recsList.some((r) => r.symbol === prev.symbol)) {
            return recsList.find((r) => r.symbol === prev.symbol) || recsList[0] || null;
          }
          return recsList[0] || null;
        });
      }
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleInspectRecommendation = (rec: StockRecommendation) => {
    setSelectedRecommendation(rec);
    const element = document.getElementById('stock-deep-dive-section');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="h-8 w-48 bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse" />
            <div className="h-4 w-64 bg-gray-200 dark:bg-gray-700 rounded mt-2 animate-pulse" />
          </div>
          <div className="h-10 w-24 bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse" />
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-white/80 dark:bg-gray-800/80 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-5">
              <div className="h-3 w-24 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
              <div className="h-8 w-32 bg-gray-200 dark:bg-gray-700 rounded mt-3 animate-pulse" />
              <div className="h-3 w-40 bg-gray-200 dark:bg-gray-700 rounded mt-2 animate-pulse" />
            </div>
          ))}
        </div>
        <div className="h-72 bg-white/80 dark:bg-gray-800/80 rounded-3xl border border-gray-200 dark:border-gray-700 animate-pulse" />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white/80 dark:bg-gray-800/80 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="h-5 w-40 bg-gray-200 dark:bg-gray-700 rounded animate-pulse mb-4" />
            <div className="space-y-4">
              {[...Array(5)].map((_, i) => (
                <div key={i} className="flex items-center justify-between">
                  <div className="h-4 w-24 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                  <div className="h-4 w-32 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                </div>
              ))}
            </div>
          </div>
          <div className="bg-white/80 dark:bg-gray-800/80 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
            <div className="h-5 w-36 bg-gray-200 dark:bg-gray-700 rounded animate-pulse mb-4" />
            <div className="grid grid-cols-2 gap-4">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="h-14 bg-gray-100 dark:bg-gray-700/50 rounded-xl animate-pulse" />
              ))}
            </div>
          </div>
        </div>
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
        return <PlusCircleIcon className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />;
      case 'PYRAMID':
        return <ArrowTrendingUpIcon className="h-4 w-4 text-blue-600 dark:text-blue-400" />;
      case 'TRAIL_SL':
        return <ShieldCheckIcon className="h-4 w-4 text-amber-600 dark:text-amber-400" />;
      case 'TARGET_HIT':
        return <FlagIcon className="h-4 w-4 text-purple-600 dark:text-purple-400" />;
      case 'CLOSED':
        return <XCircleIcon className="h-4 w-4 text-rose-600 dark:text-rose-400" />;
      default:
        return <FireIcon className="h-4 w-4 text-gray-500 dark:text-gray-400" />;
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
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-gray-900 dark:text-gray-100 tracking-tight">
            Dashboard
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Portfolio overview, algorithmic signals, and technical execution
          </p>
        </div>
        <button
          onClick={fetchData}
          className="inline-flex items-center justify-center gap-2 px-4 py-2 text-sm font-semibold text-gray-700 dark:text-gray-300 bg-white/90 dark:bg-gray-800/90 backdrop-blur-xs border border-gray-300 dark:border-gray-650 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-700 shadow-xs hover:shadow-sm transition"
        >
          <ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Data</span>
        </button>
      </div>

      {/* Motivational Quotes Banner */}
      <MotivationalQuotes />

      {/* Portfolio KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
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

      {/* Featured Stock Recommendation & Technical Analysis Deep-Dive */}
      <section id="stock-deep-dive-section" className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400">
              <SparklesIcon className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
                Featured Stock Recommendation & Technical Analysis Deep-Dive
                {recommendations.length > 0 && (
                  <span className="text-xs px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300 font-semibold hidden sm:inline-block">
                    Live Spotlight
                  </span>
                )}
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Interactive candlestick charting, institutional setup gates, and risk parameters
              </p>
            </div>
          </div>
          {recommendations.length > 0 && (
            <Link
              href="/recommendations"
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition"
            >
              <span>View all recommendations</span>
              <ArrowTopRightOnSquareIcon className="h-3.5 w-3.5" />
            </Link>
          )}
        </div>

        {/* Sleek horizontal stock selector bar / pills */}
        {recommendations.length > 0 && (
          <div className="relative">
            <div className="flex items-center gap-2.5 overflow-x-auto pb-2 pt-1 scrollbar-thin scrollbar-thumb-gray-200 dark:scrollbar-thumb-gray-700">
              {recommendations.map((rec) => {
                const isSelected = selectedRecommendation?.symbol === rec.symbol;
                const buy = rec.buy_price || 0;
                const target = rec.sell_price || (buy > 0 ? buy * 1.15 : 0);
                const upside = buy > 0 ? ((target - buy) / buy) * 100 : (rec.expected_return_percent || 15);
                const rr = rec.rr_ratio || 3.0;

                return (
                  <button
                    key={rec.symbol}
                    type="button"
                    onClick={() => setSelectedRecommendation(rec)}
                    className={`shrink-0 text-left px-3.5 py-2.5 rounded-2xl border transition-all duration-200 cursor-pointer ${
                      isSelected
                        ? 'bg-blue-50/90 dark:bg-blue-950/60 border-blue-500 dark:border-blue-500 shadow-sm ring-2 ring-blue-500/25'
                        : 'bg-white/90 dark:bg-gray-800/90 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-750'
                    }`}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-bold text-sm text-gray-900 dark:text-gray-100">
                        {cleanSymbol(rec.symbol)}
                      </span>
                      <span className="px-1.5 py-0.5 text-[10px] font-bold rounded-md bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300 uppercase">
                        {rec.recommendation_strength || 'BUY'}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 truncate max-w-[130px] mb-1.5">
                      {rec.company_name || cleanSymbol(rec.symbol)}
                    </div>
                    <div className="flex items-center gap-2 text-[11px] font-mono">
                      <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                        +{upside.toFixed(1)}%
                      </span>
                      <span className="text-gray-300 dark:text-gray-600">•</span>
                      <span className="text-gray-500 dark:text-gray-400">
                        1:{rr.toFixed(1)} R:R
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Deep Dive Component or Fallback */}
        {selectedRecommendation ? (
          <div className="rounded-3xl bg-white dark:bg-gray-850 border border-gray-200 dark:border-gray-750 p-4 sm:p-6 shadow-sm">
            <StockAnalysisDeepDive
              recommendation={selectedRecommendation}
              isModal={false}
              onSaveNote={async (newNote) => {
                if (!selectedRecommendation) return;
                await updateRecommendation(selectedRecommendation.symbol, { note: newNote });
                setSelectedRecommendation((prev) => (prev ? { ...prev, note: newNote } : null));
                setRecommendations((prev) =>
                  prev.map((r) => (r.symbol === selectedRecommendation.symbol ? { ...r, note: newNote } : r))
                );
              }}
            />
          </div>
        ) : (
          <div className="p-8 md:p-12 text-center rounded-3xl bg-white/90 dark:bg-gray-800/90 border border-dashed border-gray-300 dark:border-gray-700 space-y-3">
            <SparklesIcon className="w-10 h-10 mx-auto text-gray-400 dark:text-gray-500" />
            <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100">
              No Active Recommendations In Spotlight
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 max-w-md mx-auto">
              Run a new technical screener cycle or check the recommendations catalog to inspect deep-dive charts, institutional patterns, and trade execution plans.
            </p>
            <Link
              href="/recommendations"
              className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-xl transition shadow-xs"
            >
              View Recommendations Catalog
            </Link>
          </div>
        )}
      </section>

      {/* Day-by-Day Earnings Report */}
      <DailyEarningsChart />

      {/* Performance & Today's Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Metrics */}
        <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-md rounded-2xl shadow-xs border border-gray-200/80 dark:border-gray-700/80 p-6 hover:shadow-md transition-all">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
              Performance Metrics
            </h2>
            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Closed Trades
            </span>
          </div>
          <div className="space-y-4">
            <div className="p-3 rounded-xl bg-gray-50/70 dark:bg-gray-900/40 border border-gray-100 dark:border-gray-750">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wide">
                  Win Rate
                </span>
                <span className="text-sm font-bold font-mono text-emerald-600 dark:text-emerald-400">
                  {performance.win_rate.toFixed(1)}%
                </span>
              </div>
              <div className="w-full h-2.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-500"
                  style={{ width: `${Math.min(100, Math.max(0, performance.win_rate))}%` }}
                />
              </div>
            </div>
            <MetricRow label="Profit Factor" value={performance.profit_factor.toFixed(2)} />
            <MetricRow label="Avg Win" value={`${performance.avg_win_pct.toFixed(2)}%`} positive />
            <MetricRow label="Avg Loss" value={`${performance.avg_loss_pct.toFixed(2)}%`} negative />
            <MetricRow label="Wins / Losses" value={`${performance.wins} / ${performance.losses}`} />
            <MetricRow label="Total Closed Trades" value={String(performance.total_closed)} />
          </div>
        </div>

        {/* Today's Activity */}
        <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-md rounded-2xl shadow-xs border border-gray-200/80 dark:border-gray-700/80 p-6 hover:shadow-md transition-all">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
              Today's Activity
            </h2>
            <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Execution Events
            </span>
          </div>
          <div className="grid grid-cols-2 gap-3.5">
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
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-750 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-750 bg-gradient-to-r from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 flex items-center justify-between">
            <h2 className="text-base font-bold text-gray-900 dark:text-gray-100 flex items-center gap-2">
              <span>Open Positions</span>
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300">
                {positions.length} Active
              </span>
            </h2>
            <Link
              href="/positions"
              className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700"
            >
              <span>Manage all</span>
              <ArrowTopRightOnSquareIcon className="h-3.5 w-3.5" />
            </Link>
          </div>

          {/* Desktop Table View */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50/60 dark:bg-gray-900/60 text-gray-500 dark:text-gray-400">
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider">Symbol</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider">Why Picked (Setup)</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider">Qty</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider">Entry</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider">Current</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider">P&L</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider">Investment</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-750">
                {positions.map((pos) => (
                  <tr key={pos.symbol} className="hover:bg-gray-50/60 dark:hover:bg-gray-750/30 transition-colors">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-gray-900 dark:text-gray-100">{cleanSymbol(pos.symbol)}</span>
                        {pos.adds_count > 0 && (
                          <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded">
                            +{pos.adds_count}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <StockPickNoteTag
                        symbol={cleanSymbol(pos.symbol)}
                        strategy={pos.strategy}
                        reason={pos.reason}
                        note={pos.note}
                        entryPrice={pos.entry_price}
                        stopLoss={pos.stop_loss}
                        target={pos.target}
                        entryAtr={pos.entry_atr}
                        onSaveNote={async (newNote) => {
                          await updatePosition(pos.symbol, { note: newNote });
                          fetchData();
                        }}
                      />
                    </td>
                    <td className="px-4 py-3 text-right font-medium text-gray-900 dark:text-gray-100">{pos.quantity}</td>
                    <td className="px-4 py-3 text-right text-gray-600 dark:text-gray-400">₹{pos.entry_price.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right text-gray-600 dark:text-gray-400">₹{pos.current_price.toFixed(2)}</td>
                    <td className="px-4 py-3 text-right">
                      <div className={`font-semibold ${pos.pnl_pct >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                        {pos.pnl_pct >= 0 ? '+' : ''}{pos.pnl_pct.toFixed(2)}%
                      </div>
                      <div className={`text-xs ${pos.unrealized_pnl >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
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

          {/* Mobile Card Stack */}
          <div className="block md:hidden divide-y divide-gray-100 dark:divide-gray-750">
            {positions.map((pos) => {
              const isProfitable = pos.pnl_pct >= 0;
              return (
                <div key={pos.symbol} className="p-4 space-y-3 hover:bg-gray-50/50 dark:hover:bg-gray-750/30 transition">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-base text-gray-900 dark:text-gray-100">
                          {cleanSymbol(pos.symbol)}
                        </span>
                        {pos.adds_count > 0 && (
                          <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 rounded">
                            +{pos.adds_count}
                          </span>
                        )}
                      </div>
                      <div className="mt-1">
                        <span className="px-2 py-0.5 text-[11px] font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded-md">
                          {pos.strategy || 'Swing_v3'}
                        </span>
                      </div>
                    </div>
                    <div className="text-right">
                      <span
                        className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold ${
                          isProfitable
                            ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300'
                            : 'bg-rose-100 dark:bg-rose-950/60 text-rose-700 dark:text-rose-300'
                        }`}
                      >
                        {isProfitable ? '+' : ''}{pos.pnl_pct.toFixed(2)}%
                      </span>
                      <div
                        className={`text-xs font-mono font-medium mt-0.5 ${
                          isProfitable ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
                        }`}
                      >
                        {pos.unrealized_pnl >= 0 ? '+' : ''}₹{pos.unrealized_pnl.toLocaleString('en-IN')}
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2 p-2.5 rounded-xl bg-gray-50 dark:bg-gray-900/60 text-xs">
                    <div>
                      <span className="text-[10px] text-gray-400 block uppercase">Entry</span>
                      <span className="font-mono font-medium text-gray-800 dark:text-gray-200">₹{pos.entry_price.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-gray-400 block uppercase">Current</span>
                      <span className="font-mono font-medium text-gray-800 dark:text-gray-200">₹{pos.current_price.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-[10px] text-gray-400 block uppercase">Invested</span>
                      <span className="font-mono font-medium text-gray-800 dark:text-gray-200">₹{pos.total_investment.toLocaleString('en-IN')}</span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-xs pt-1">
                    <span className="text-gray-500 dark:text-gray-400 font-mono">Qty: {pos.quantity}</span>
                    <StockPickNoteTag
                      symbol={cleanSymbol(pos.symbol)}
                      strategy={pos.strategy}
                      reason={pos.reason}
                      note={pos.note}
                      entryPrice={pos.entry_price}
                      stopLoss={pos.stop_loss}
                      target={pos.target}
                      entryAtr={pos.entry_atr}
                      onSaveNote={async (newNote) => {
                        await updatePosition(pos.symbol, { note: newNote });
                        fetchData();
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Latest Stock Recommendations */}
      {recommendations.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-750 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-750 bg-gradient-to-r from-gray-50 to-white dark:from-gray-900 dark:to-gray-800 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <SparklesIcon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              <h2 className="text-base font-bold text-gray-900 dark:text-gray-100">
                Latest Stock Recommendations
                <span className="ml-2 text-xs font-normal text-gray-500 dark:text-gray-400">
                  (Showing {recommendations.length})
                </span>
              </h2>
            </div>
            <Link
              href="/recommendations"
              className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition"
            >
              <span>View catalog</span>
              <ArrowTopRightOnSquareIcon className="h-3.5 w-3.5" />
            </Link>
          </div>

          {/* Desktop Table View */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50/60 dark:bg-gray-900/60 text-gray-500 dark:text-gray-400">
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider">Stock</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider">Why Picked (Setup Context)</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider">Confidence</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider">Buy Price</th>
                  <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider">Target</th>
                  <th className="px-4 py-3 text-center text-xs font-semibold uppercase tracking-wider">Strength</th>
                  <th className="px-4 py-3 text-center text-xs font-semibold uppercase tracking-wider">Spotlight</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-750">
                {recommendations.map((rec) => {
                  const isSelected = selectedRecommendation?.symbol === rec.symbol;
                  return (
                    <tr
                      key={rec.symbol}
                      className={`transition-colors ${
                        isSelected ? 'bg-blue-50/40 dark:bg-blue-950/20' : 'hover:bg-gray-50/60 dark:hover:bg-gray-750/30'
                      }`}
                    >
                      <td className="px-4 py-3">
                        <div>
                          <span className="font-bold text-gray-900 dark:text-gray-100">{cleanSymbol(rec.symbol)}</span>
                          {rec.company_name && (
                            <div className="text-xs text-gray-500 dark:text-gray-400 truncate max-w-[160px]">{rec.company_name}</div>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <StockPickNoteTag
                          symbol={cleanSymbol(rec.symbol)}
                          strategy={rec.strategy_name || 'Swing_Trading_v3'}
                          reason={rec.reason}
                          note={rec.note}
                          technicalScore={rec.technical_score}
                          fundamentalScore={rec.fundamental_score}
                          sentimentScore={rec.sentiment_score}
                          combinedScore={rec.combined_score}
                          compositeScore={rec.composite_score}
                          entryPrice={rec.buy_price}
                          stopLoss={rec.stop_loss || (rec.buy_price ? rec.buy_price * 0.94 : undefined)}
                          target={rec.sell_price}
                          entryAtr={rec.entry_atr}
                          rrRatio={rec.rr_ratio || 3.0}
                          suggestedQuantity={rec.suggested_quantity}
                          allocationPct={rec.allocation_pct}
                          recommendationStrength={rec.recommendation_strength}
                          patternDetected={rec.pattern_detected}
                          setupChecklist={rec.setup_checklist}
                          technicalPillars={rec.technical_pillars}
                          recommendation={rec}
                          onSaveNote={async (newNote) => {
                            await updateRecommendation(rec.symbol, { note: newNote });
                            if (selectedRecommendation?.symbol === rec.symbol) {
                              setSelectedRecommendation((prev) => (prev ? { ...prev, note: newNote } : null));
                            }
                            fetchData();
                          }}
                        />
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className="font-mono font-bold text-blue-600 dark:text-blue-400">
                          {((rec.combined_score || 0) * 100).toFixed(0)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-medium text-gray-900 dark:text-gray-100">
                        {rec.buy_price ? `₹${rec.buy_price.toFixed(2)}` : 'Market'}
                      </td>
                      <td className="px-4 py-3 text-right font-medium text-emerald-600 dark:text-emerald-400">
                        {rec.sell_price ? `₹${rec.sell_price.toFixed(2)}` : 'ATR Multi-Target'}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className="px-2.5 py-1 text-xs font-bold rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
                          {rec.recommendation_strength || 'BUY'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <button
                          type="button"
                          onClick={() => handleInspectRecommendation(rec)}
                          className={`inline-flex items-center gap-1 px-3 py-1 text-xs font-semibold rounded-lg transition cursor-pointer ${
                            isSelected
                              ? 'bg-blue-600 text-white shadow-xs'
                              : 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 hover:bg-blue-100 dark:hover:bg-blue-900/50'
                          }`}
                        >
                          <EyeIcon className="w-3.5 h-3.5" />
                          <span>{isSelected ? 'Active' : 'Inspect'}</span>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Mobile Card Stack */}
          <div className="block md:hidden divide-y divide-gray-100 dark:divide-gray-750">
            {recommendations.map((rec) => {
              const isSelected = selectedRecommendation?.symbol === rec.symbol;
              const confidencePct = Math.round((rec.combined_score || 0) * 100);
              return (
                <div
                  key={rec.symbol}
                  className={`p-4 space-y-3 transition ${
                    isSelected ? 'bg-blue-50/50 dark:bg-blue-950/30' : 'hover:bg-gray-50/50 dark:hover:bg-gray-750/30'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-base text-gray-900 dark:text-gray-100">
                          {cleanSymbol(rec.symbol)}
                        </span>
                        <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
                          {rec.recommendation_strength || 'BUY'}
                        </span>
                      </div>
                      {rec.company_name && (
                        <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 truncate max-w-[200px]">
                          {rec.company_name}
                        </p>
                      )}
                    </div>
                    <div className="text-right">
                      <span className="text-[10px] text-gray-400 uppercase block font-semibold">Confidence</span>
                      <span className="font-mono text-sm font-bold text-blue-600 dark:text-blue-400">
                        {confidencePct}%
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 p-2.5 rounded-xl bg-gray-50 dark:bg-gray-900/60 text-xs">
                    <div>
                      <span className="text-[10px] text-gray-400 block uppercase">Buy Price</span>
                      <span className="font-mono font-medium text-gray-800 dark:text-gray-200">
                        {rec.buy_price ? `₹${rec.buy_price.toFixed(2)}` : 'Market'}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-gray-400 block uppercase">Target Price</span>
                      <span className="font-mono font-medium text-emerald-600 dark:text-emerald-400">
                        {rec.sell_price ? `₹${rec.sell_price.toFixed(2)}` : 'ATR Multi-Target'}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between gap-2 pt-1">
                    <StockPickNoteTag
                      symbol={cleanSymbol(rec.symbol)}
                      strategy={rec.strategy_name || 'Swing_Trading_v3'}
                      reason={rec.reason}
                      note={rec.note}
                      technicalScore={rec.technical_score}
                      fundamentalScore={rec.fundamental_score}
                      sentimentScore={rec.sentiment_score}
                      combinedScore={rec.combined_score}
                      compositeScore={rec.composite_score}
                      entryPrice={rec.buy_price}
                      stopLoss={rec.stop_loss || (rec.buy_price ? rec.buy_price * 0.94 : undefined)}
                      target={rec.sell_price}
                      entryAtr={rec.entry_atr}
                      rrRatio={rec.rr_ratio || 3.0}
                      suggestedQuantity={rec.suggested_quantity}
                      allocationPct={rec.allocation_pct}
                      recommendationStrength={rec.recommendation_strength}
                      patternDetected={rec.pattern_detected}
                      setupChecklist={rec.setup_checklist}
                      technicalPillars={rec.technical_pillars}
                      recommendation={rec}
                      onSaveNote={async (newNote) => {
                        await updateRecommendation(rec.symbol, { note: newNote });
                        if (selectedRecommendation?.symbol === rec.symbol) {
                          setSelectedRecommendation((prev) => (prev ? { ...prev, note: newNote } : null));
                        }
                        fetchData();
                      }}
                    />

                    <button
                      type="button"
                      onClick={() => handleInspectRecommendation(rec)}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl transition cursor-pointer ${
                        isSelected
                          ? 'bg-blue-600 text-white shadow-xs'
                          : 'bg-blue-50 dark:bg-blue-900/40 text-blue-600 dark:text-blue-300 hover:bg-blue-100'
                      }`}
                    >
                      <SparklesIcon className="w-3.5 h-3.5" />
                      <span>{isSelected ? 'Viewing Chart' : 'Inspect Chart'}</span>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Activity Feed */}
      {activityFeed.length > 0 && (
        <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-md rounded-2xl shadow-xs border border-gray-200/80 dark:border-gray-700/80 p-6 hover:shadow-md transition-all">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
              Recent Activity
            </h2>
            <Link
              href="/activity-log"
              className="inline-flex items-center gap-1 text-xs font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700"
            >
              <span>Full audit log</span>
              <ArrowTopRightOnSquareIcon className="h-3.5 w-3.5" />
            </Link>
          </div>
          <div className="space-y-3">
            {activityFeed.slice(0, 8).map((log, idx) => (
              <div key={idx} className="flex items-center gap-3 p-3 bg-gray-50/80 dark:bg-gray-900/50 rounded-xl border border-gray-100/60 dark:border-gray-750">
                <div className="p-1.5 rounded-lg bg-white dark:bg-gray-800 shadow-xs">
                  {getActionIcon(log.action)}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-xs sm:text-sm text-gray-900 dark:text-gray-100">{log.symbol}</span>
                    <span className="text-[11px] font-medium text-gray-500 dark:text-gray-400">{log.action}</span>
                  </div>
                </div>
                <span className="text-[11px] text-gray-400 dark:text-gray-500 whitespace-nowrap font-mono">
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
    <div className="group relative bg-white/90 dark:bg-gray-800/90 backdrop-blur-md rounded-2xl p-5 border border-gray-200/80 dark:border-gray-700/80 hover:border-blue-500/40 dark:hover:border-blue-400/40 shadow-xs hover:shadow-md transition-all duration-200 hover:-translate-y-0.5">
      <div className="flex items-start justify-between">
        <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
          {label}
        </p>
        {Icon && (
          <div
            className={`p-2.5 rounded-xl transition-colors ${
              positive !== undefined
                ? positive
                  ? 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400'
                  : 'bg-rose-50 dark:bg-rose-950/40 text-rose-600 dark:text-rose-400'
                : 'bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400'
            }`}
          >
            <Icon className="h-5 w-5" />
          </div>
        )}
      </div>
      <p
        className={`text-2xl font-extrabold font-mono mt-2 tracking-tight ${
          positive !== undefined
            ? positive
              ? 'text-emerald-600 dark:text-emerald-400'
              : 'text-rose-600 dark:text-rose-400'
            : 'text-gray-900 dark:text-gray-100'
        }`}
      >
        {value}
      </p>
      {subtitle && (
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1.5">{subtitle}</p>
      )}
    </div>
  );
}

function MetricRow({ label, value, positive, negative }: { label: string; value: string; positive?: boolean; negative?: boolean }) {
  return (
    <div className="flex items-center justify-between p-2 rounded-lg hover:bg-gray-50/60 dark:hover:bg-gray-750/30 transition-colors">
      <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">{label}</span>
      <span
        className={`text-xs sm:text-sm font-bold font-mono ${
          positive
            ? 'text-emerald-600 dark:text-emerald-400'
            : negative
            ? 'text-rose-600 dark:text-rose-400'
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
    green: 'bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400',
    blue: 'bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400',
    amber: 'bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400',
    purple: 'bg-purple-50 dark:bg-purple-950/40 text-purple-600 dark:text-purple-400',
    red: 'bg-rose-50 dark:bg-rose-950/40 text-rose-600 dark:text-rose-400',
  };

  return (
    <div className="flex items-center gap-3 p-3 bg-gray-50/70 dark:bg-gray-900/50 rounded-xl border border-gray-100/60 dark:border-gray-750">
      <div className={`p-2.5 rounded-xl ${colorClasses[color as keyof typeof colorClasses]}`}>
        <Icon className="h-4.5 w-4.5" />
      </div>
      <div>
        <p className="text-[11px] text-gray-500 dark:text-gray-400 font-medium">{label}</p>
        <p className="text-base sm:text-lg font-bold font-mono text-gray-900 dark:text-gray-100">{value}</p>
      </div>
    </div>
  );
}
