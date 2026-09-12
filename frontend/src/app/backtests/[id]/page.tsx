'use client';

import { useState, useEffect, useMemo, Suspense } from 'react';
import { useParams, useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import {
  ArrowLeftIcon,
  ChartBarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ShieldCheckIcon,
  ClockIcon,
  ArrowPathIcon,
  FunnelIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
  InformationCircleIcon,
  CurrencyRupeeIcon,
  SparklesIcon,
  ChevronUpDownIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ArrowsPointingOutIcon,
  CheckCircleIcon,
  XCircleIcon,
  TableCellsIcon,
  Square3Stack3DIcon,
  Cog6ToothIcon,
  CalendarDaysIcon,
  TrophyIcon,
  BoltIcon,
} from '@heroicons/react/24/outline';
import {
  getBacktestSession,
  getBacktestTrades,
  getBacktestSymbols,
  getBacktestYearlyBreakdown,
  BacktestSession,
  BacktestTrade,
  BacktestSymbolSummary,
  BacktestYearlyBreakdownItem,
} from '@/lib/api';
import YearlyBreakdownTable from '@/components/backtests/YearlyBreakdownTable';
import UltimatePhasesView from '@/components/backtests/UltimatePhasesView';

function BacktestDetailPageContent() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionId = params?.id as string;

  const [session, setSession] = useState<BacktestSession | null>(null);
  const [symbolsSummary, setSymbolsSummary] = useState<BacktestSymbolSummary[]>([]);
  const [yearlyBreakdown, setYearlyBreakdown] = useState<BacktestYearlyBreakdownItem[]>([]);
  const [loadingSession, setLoadingSession] = useState(true);

  // Journal Trades State
  const [trades, setTrades] = useState<BacktestTrade[]>([]);
  const [totalTrades, setTotalTrades] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [limit, setLimit] = useState(50);
  const [loadingTrades, setLoadingTrades] = useState(false);

  // Filters State
  const [activeTab, setActiveTab] = useState<'journal' | 'ultimate' | 'yearly' | 'symbols' | 'config'>('journal');
  const [search, setSearch] = useState('');
  const [tradeType, setTradeType] = useState('ALL');
  const [exitReason, setExitReason] = useState('ALL');
  const [pattern, setPattern] = useState('ALL');
  const [outcome, setOutcome] = useState('ALL');
  const [sortBy, setSortBy] = useState('entry_date');
  const [sortDir, setSortDir] = useState<'desc' | 'asc'>('desc');

  // URL query sync
  useEffect(() => {
    const tabParam = searchParams?.get('tab');
    if (
      tabParam === 'ultimate' ||
      tabParam === 'yearly' ||
      tabParam === 'symbols' ||
      tabParam === 'config' ||
      tabParam === 'journal'
    ) {
      setActiveTab(tabParam as any);
    }
  }, [searchParams]);

  // By Symbol Sorting State
  const [symSortBy, setSymSortBy] = useState<
    'symbol' | 'total_pnl' | 'win_rate' | 'winning_exits' | 'buys' | 'pyramids' | 'sells' | 'total_events'
  >('total_pnl');
  const [symSortDir, setSymSortDir] = useState<'desc' | 'asc'>('desc');

  // Selected trade for detail modal
  const [selectedTrade, setSelectedTrade] = useState<BacktestTrade | null>(null);

  // Journal Sort Click Handler
  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortDir((prev) => (prev === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortBy(field);
      setSortDir(field === 'symbol' || field === 'trade_type' ? 'asc' : 'desc');
    }
    setPage(1);
  };

  const renderSortIcon = (field: string) => {
    if (sortBy === field) {
      return sortDir === 'asc' ? (
        <ChevronUpIcon className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400 shrink-0" />
      ) : (
        <ChevronDownIcon className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400 shrink-0" />
      );
    }
    return (
      <ChevronUpDownIcon className="w-3.5 h-3.5 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
    );
  };

  // By Symbol Sort Click Handler
  const handleSymSort = (field: typeof symSortBy) => {
    if (symSortBy === field) {
      setSymSortDir((prev) => (prev === 'desc' ? 'asc' : 'desc'));
    } else {
      setSymSortBy(field);
      setSymSortDir(field === 'symbol' ? 'asc' : 'desc');
    }
  };

  const renderSymSortIcon = (field: typeof symSortBy) => {
    if (symSortBy === field) {
      return symSortDir === 'asc' ? (
        <ChevronUpIcon className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400 shrink-0" />
      ) : (
        <ChevronDownIcon className="w-3.5 h-3.5 text-indigo-600 dark:text-indigo-400 shrink-0" />
      );
    }
    return (
      <ChevronUpDownIcon className="w-3.5 h-3.5 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
    );
  };

  const sortedSymbols = useMemo(() => {
    return [...symbolsSummary].sort((a, b) => {
      const valA = a[symSortBy] ?? 0;
      const valB = b[symSortBy] ?? 0;
      if (typeof valA === 'string' && typeof valB === 'string') {
        return symSortDir === 'asc' ? valA.localeCompare(valB) : valB.localeCompare(valA);
      }
      return symSortDir === 'asc' ? Number(valA) - Number(valB) : Number(valB) - Number(valA);
    });
  }, [symbolsSummary, symSortBy, symSortDir]);


  // Load Session Info & Symbols
  const fetchSessionInfo = async () => {
    if (!sessionId) return;
    setLoadingSession(true);
    try {
      const [sessRes, symRes, yearlyRes] = await Promise.all([
        getBacktestSession(sessionId),
        getBacktestSymbols(sessionId),
        getBacktestYearlyBreakdown(sessionId),
      ]);
      if (sessRes.status === 'success' && sessRes.session) {
        setSession(sessRes.session);
        if (sessRes.session.yearly_breakdown && sessRes.session.yearly_breakdown.length > 0) {
          setYearlyBreakdown(sessRes.session.yearly_breakdown);
        }
      }
      if (symRes.status === 'success' && symRes.symbols) {
        setSymbolsSummary(symRes.symbols);
      }
      if (yearlyRes.status === 'success' && yearlyRes.breakdown && yearlyRes.breakdown.length > 0) {
        setYearlyBreakdown(yearlyRes.breakdown);
      }
    } catch (err) {
      console.error('Failed to load session details:', err);
    } finally {
      setLoadingSession(false);
    }
  };

  // Load Journal Trades with Filters
  const fetchTrades = async () => {
    if (!sessionId) return;
    setLoadingTrades(true);
    try {
      const res = await getBacktestTrades(sessionId, {
        search: search.trim() || undefined,
        trade_type: tradeType !== 'ALL' ? tradeType : undefined,
        exit_reason: exitReason !== 'ALL' ? exitReason : undefined,
        pattern: pattern !== 'ALL' ? pattern : undefined,
        outcome: outcome !== 'ALL' ? outcome : undefined,
        page,
        limit,
        sort_by: sortBy,
        sort_dir: sortDir,
      });

      if (res.status === 'success') {
        setTrades(res.trades || []);
        setTotalTrades(res.total || 0);
        setTotalPages(res.total_pages || 1);
      }
    } catch (err) {
      console.error('Failed to load trades:', err);
    } finally {
      setLoadingTrades(false);
    }
  };

  useEffect(() => {
    fetchSessionInfo();
  }, [sessionId]);

  useEffect(() => {
    fetchTrades();
  }, [sessionId, page, limit, tradeType, exitReason, pattern, outcome, sortBy, sortDir]);

  // Debounced search
  useEffect(() => {
    const handler = setTimeout(() => {
      setPage(1);
      fetchTrades();
    }, 350);
    return () => clearTimeout(handler);
  }, [search]);

  // Distinct exit reasons and patterns from session execution summary
  const availableReasons = useMemo(() => {
    if (!session?.execution_summary?.exit_reasons) return [];
    return Object.keys(session.execution_summary.exit_reasons);
  }, [session]);

  const availablePatterns = useMemo(() => {
    if (!session?.execution_summary?.entry_patterns) return [];
    return Object.keys(session.execution_summary.entry_patterns);
  }, [session]);

  const metrics = session?.summary_metrics || {};
  const isUltimateSession =
    session?.session_type === 'ultimate' ||
    session?.session_name?.toLowerCase().includes('ultimate') ||
    !!session?.ultimate_phases;

  return (
    <div className="space-y-6 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Top Breadcrumb & Actions Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 pb-4 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center gap-3">
          <Link
            href="/backtests"
            className="p-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors shadow-sm"
            title="Back to Backtests"
          >
            <ArrowLeftIcon className="w-5 h-5" />
          </Link>

          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h1 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">
                {session?.strategy_name || 'Backtest Session'}
              </h1>
              {isUltimateSession ? (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-bold bg-purple-500/15 text-purple-700 dark:text-purple-300 border border-purple-500/30">
                  <TrophyIcon className="w-3.5 h-3.5 text-purple-500" />
                  Ultimate Backtest (6-Phase)
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 border border-indigo-500/30">
                  <BoltIcon className="w-3.5 h-3.5 text-indigo-500" />
                  Portfolio Backtest
                </span>
              )}
              <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800">
                {session?.status || 'Completed'}
              </span>
            </div>
            <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400 flex items-center gap-2 mt-0.5">
              <span>{session?.date_range?.start_date || 'N/A'} → {session?.date_range?.end_date || 'N/A'}</span>
              <span>•</span>
              <span>{session?.execution_summary?.total_events || 0} Execution Actions</span>
              <span>•</span>
              <span>{session?.execution_summary?.unique_symbols_traded || 0} Traded Stocks</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              fetchSessionInfo();
              fetchTrades();
            }}
            className="inline-flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors shadow-sm"
          >
            <ArrowPathIcon className={`w-4 h-4 ${loadingTrades ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 sm:gap-4">
        {/* Total Return */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between text-gray-500 dark:text-gray-400 text-xs font-medium">
            <span>Total Return</span>
            {(metrics.total_return_pct ?? 0) >= 0 ? (
              <ArrowTrendingUpIcon className="w-4 h-4 text-emerald-500" />
            ) : (
              <ArrowTrendingDownIcon className="w-4 h-4 text-rose-500" />
            )}
          </div>
          <div
            className={`text-xl sm:text-2xl font-bold mt-1.5 ${
              (metrics.total_return_pct ?? 0) >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
            }`}
          >
            {(metrics.total_return_pct ?? 0) >= 0 ? '+' : ''}
            {Number(metrics.total_return_pct ?? 0).toFixed(1)}%
          </div>
          <div className="text-[11px] text-gray-400 mt-1">
            ₹{Number(metrics.initial_capital ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })} → ₹{Number(metrics.final_portfolio_value ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
          </div>
        </div>

        {/* CAGR */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <span className="text-gray-500 dark:text-gray-400 text-xs font-medium block">CAGR (Annualized)</span>
          <div className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white mt-1.5">
            {Number(metrics.cagr ?? 0).toFixed(2)}%
          </div>
          <div className="text-[11px] text-gray-400 mt-1">
            Compounded growth rate
          </div>
        </div>

        {/* Max Drawdown */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <span className="text-gray-500 dark:text-gray-400 text-xs font-medium block">Max Drawdown</span>
          <div className="text-xl sm:text-2xl font-bold text-amber-600 dark:text-amber-400 mt-1.5">
            {Number(metrics.max_drawdown_pct ?? 0).toFixed(1)}%
          </div>
          <div className="text-[11px] text-gray-400 mt-1">
            Deepest peak-to-trough
          </div>
        </div>

        {/* Sharpe Ratio */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <span className="text-gray-500 dark:text-gray-400 text-xs font-medium block">Sharpe Ratio</span>
          <div className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white mt-1.5">
            {Number(metrics.sharpe_ratio ?? 0).toFixed(2)}
          </div>
          <div className="text-[11px] text-gray-400 mt-1">
            Risk-adjusted return
          </div>
        </div>

        {/* Win Rate */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <span className="text-gray-500 dark:text-gray-400 text-xs font-medium block">Win Rate</span>
          <div className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white mt-1.5">
            {Number(metrics.win_rate ?? 0).toFixed(1)}%
          </div>
          <div className="text-[11px] text-gray-400 mt-1">
            {metrics.total_trades ?? 0} round-trip trades
          </div>
        </div>

        {/* Profit Factor */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-4 shadow-sm border border-gray-200 dark:border-gray-700">
          <span className="text-gray-500 dark:text-gray-400 text-xs font-medium block">Profit Factor</span>
          <div className="text-xl sm:text-2xl font-bold text-emerald-600 dark:text-emerald-400 mt-1.5">
            {Number(metrics.profit_factor ?? 0).toFixed(2)}x
          </div>
          <div className="text-[11px] text-gray-400 mt-1">
            Exp: ₹{Number(metrics.expectancy ?? 0).toFixed(0)}/trade
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center gap-2 border-b border-gray-200 dark:border-gray-700 overflow-x-auto">
        {isUltimateSession && (
          <button
            onClick={() => setActiveTab('ultimate')}
            className={`inline-flex items-center gap-2 py-3 px-4 text-sm font-semibold border-b-2 transition-colors shrink-0 ${
              activeTab === 'ultimate'
                ? 'border-purple-600 text-purple-600 dark:border-purple-400 dark:text-purple-400'
                : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
            }`}
          >
            <TrophyIcon className="w-4 h-4 text-purple-500" />
            🏆 Ultimate Analysis (6-Phase)
          </button>
        )}

        <button
          onClick={() => setActiveTab('journal')}
          className={`inline-flex items-center gap-2 py-3 px-4 text-sm font-semibold border-b-2 transition-colors shrink-0 ${
            activeTab === 'journal'
              ? 'border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
        >
          <TableCellsIcon className="w-4 h-4" />
          Trade Journal & Actions ({totalTrades})
        </button>

        <button
          onClick={() => setActiveTab('yearly')}
          className={`inline-flex items-center gap-2 py-3 px-4 text-sm font-semibold border-b-2 transition-colors shrink-0 ${
            activeTab === 'yearly'
              ? 'border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
        >
          <CalendarDaysIcon className="w-4 h-4" />
          YoY Performance {yearlyBreakdown.length > 0 && `(${yearlyBreakdown.length} Years)`}
        </button>

        <button
          onClick={() => setActiveTab('symbols')}
          className={`inline-flex items-center gap-2 py-3 px-4 text-sm font-semibold border-b-2 transition-colors shrink-0 ${
            activeTab === 'symbols'
              ? 'border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
        >
          <Square3Stack3DIcon className="w-4 h-4" />
          By Symbol ({symbolsSummary.length})
        </button>

        <button
          onClick={() => setActiveTab('config')}
          className={`inline-flex items-center gap-2 py-3 px-4 text-sm font-semibold border-b-2 transition-colors shrink-0 ${
            activeTab === 'config'
              ? 'border-indigo-600 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400'
              : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
          }`}
        >
          <Cog6ToothIcon className="w-4 h-4" />
          Strategy & Capital Rules
        </button>
      </div>

      {/* TAB 0: ULTIMATE ANALYSIS (6-PHASE ROBUSTNESS MATRIX) */}
      {activeTab === 'ultimate' && (
        session?.ultimate_phases ? (
          <UltimatePhasesView phases={session.ultimate_phases} strategyName={session.strategy_name} />
        ) : (
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-12 text-center border border-gray-200 dark:border-gray-700 space-y-3">
            <TrophyIcon className="w-12 h-12 mx-auto text-purple-400" />
            <h3 className="text-lg font-bold text-gray-900 dark:text-white">6-Phase Statistical Data</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 max-w-md mx-auto">
              This session was run as an ultimate backtest. Detailed statistical verification, stress tests, and confidence scores are attached to this session.
            </p>
          </div>
        )
      )}

      {/* TAB 1: DETAILED TRADE JOURNAL */}
      {activeTab === 'journal' && (
        <div className="space-y-4">
          {/* Filter Toolbar */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-4 shadow-sm border border-gray-200 dark:border-gray-700 flex flex-wrap items-center gap-3">
            {/* Search Input */}
            <div className="relative flex-1 min-w-[200px]">
              <MagnifyingGlassIcon className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search symbol (e.g. HAL, GMMPFAUDLR)..."
                className="w-full pl-9 pr-8 py-2 text-sm rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              {search && (
                <button
                  onClick={() => setSearch('')}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  <XMarkIcon className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* Action Type Filter */}
            <select
              value={tradeType}
              onChange={(e) => {
                setTradeType(e.target.value);
                setPage(1);
              }}
              className="py-2 px-3 text-sm rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="ALL">All Actions</option>
              <option value="BUY">🟢 Initial Buy (Entry)</option>
              <option value="PYRAMID_ADD">🔵 Pyramid Add (Scale In)</option>
              <option value="PARTIAL_SELL">🟡 Target Trim (Partial Sell)</option>
              <option value="SELL">🔴 Full Exit (Sell)</option>
            </select>

            {/* Outcome Filter */}
            <select
              value={outcome}
              onChange={(e) => {
                setOutcome(e.target.value);
                setPage(1);
              }}
              className="py-2 px-3 text-sm rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              <option value="ALL">All Outcomes</option>
              <option value="profit">Wins (P&L &gt; 0)</option>
              <option value="loss">Losses (P&L &lt; 0)</option>
            </select>

            {/* Exit Reason Filter */}
            {availableReasons.length > 0 && (
              <select
                value={exitReason}
                onChange={(e) => {
                  setExitReason(e.target.value);
                  setPage(1);
                }}
                className="py-2 px-3 text-sm rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="ALL">All Exit Reasons</option>
                {availableReasons.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            )}

            {/* Entry Pattern Filter */}
            {availablePatterns.length > 0 && (
              <select
                value={pattern}
                onChange={(e) => {
                  setPattern(e.target.value);
                  setPage(1);
                }}
                className="py-2 px-3 text-sm rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-800 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="ALL">All Patterns</option>
                {availablePatterns.map((p) => (
                  <option key={p} value={p}>
                    {p.replace(/_/g, ' ')}
                  </option>
                ))}
              </select>
            )}

            {/* Sort direction */}
            <button
              onClick={() => {
                setSortDir(sortDir === 'desc' ? 'asc' : 'desc');
                setPage(1);
              }}
              className="p-2 rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
              title={`Sort ${sortDir === 'desc' ? 'Ascending' : 'Descending'}`}
            >
              <ChevronUpDownIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Journal Table */}
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
            {loadingTrades ? (
              <div className="p-12 text-center text-gray-500 dark:text-gray-400">
                <ArrowPathIcon className="w-8 h-8 animate-spin mx-auto mb-3 text-indigo-500" />
                <p>Loading trade journal...</p>
              </div>
            ) : trades.length === 0 ? (
              <div className="p-12 text-center text-gray-500 dark:text-gray-400">
                <InformationCircleIcon className="w-10 h-10 mx-auto text-gray-400 mb-2" />
                <p className="font-semibold text-gray-700 dark:text-gray-300">No Trades Match Selected Filter</p>
                <p className="text-sm">Try clearing filters or changing search keywords.</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="bg-gray-50 dark:bg-gray-900 text-gray-600 dark:text-gray-300 uppercase text-xs tracking-wider border-b border-gray-200 dark:border-gray-700 font-semibold">
                    <tr>
                      <th
                        onClick={() => handleSort('entry_date')}
                        className="py-3.5 px-4 sm:px-6 cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                        title="Click to sort by Date"
                      >
                        <div className="flex items-center gap-1.5">
                          <span>Date</span>
                          {renderSortIcon('entry_date')}
                        </div>
                      </th>
                      <th
                        onClick={() => handleSort('symbol')}
                        className="py-3.5 px-4 cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                        title="Click to sort by Stock Symbol"
                      >
                        <div className="flex items-center gap-1.5">
                          <span>Symbol</span>
                          {renderSortIcon('symbol')}
                        </div>
                      </th>
                      <th
                        onClick={() => handleSort('trade_type')}
                        className="py-3.5 px-4 cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                        title="Click to sort by Action Type"
                      >
                        <div className="flex items-center gap-1.5">
                          <span>Action</span>
                          {renderSortIcon('trade_type')}
                        </div>
                      </th>
                      <th
                        onClick={() => handleSort('entry_price')}
                        className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                        title="Click to sort by Fill Price"
                      >
                        <div className="flex items-center justify-end gap-1.5">
                          <span>Fill Price</span>
                          {renderSortIcon('entry_price')}
                        </div>
                      </th>
                      <th
                        onClick={() => handleSort('position_value')}
                        className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                        title="Click to sort by Position Size"
                      >
                        <div className="flex items-center justify-end gap-1.5">
                          <span>Qty & Size</span>
                          {renderSortIcon('position_value')}
                        </div>
                      </th>
                      <th
                        onClick={() => handleSort('stop_loss')}
                        className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                        title="Click to sort by Stop Loss"
                      >
                        <div className="flex items-center justify-end gap-1.5">
                          <span>SL / Target</span>
                          {renderSortIcon('stop_loss')}
                        </div>
                      </th>
                      <th
                        onClick={() => handleSort('pnl')}
                        className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                        title="Click to sort by Realized P&L"
                      >
                        <div className="flex items-center justify-end gap-1.5">
                          <span>P&L (₹ / %)</span>
                          {renderSortIcon('pnl')}
                        </div>
                      </th>
                      <th
                        onClick={() => handleSort('entry_pattern')}
                        className="py-3.5 px-4 cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                        title="Click to sort by Technical Pattern / Reason"
                      >
                        <div className="flex items-center gap-1.5">
                          <span>Pattern / Reason</span>
                          {renderSortIcon('entry_pattern')}
                        </div>
                      </th>
                      <th className="py-3.5 px-4 text-center">Inspect</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 dark:divide-gray-700/80">
                    {trades.map((t) => {
                      const isSell = t.trade_type === 'SELL' || t.trade_type === 'PARTIAL_SELL';
                      const pnl = t.pnl ?? 0;
                      const hasPnl = isSell && pnl !== 0;
                      const isProfit = pnl > 0;
                      const displayDate = isSell && t.exit_date ? t.exit_date : t.entry_date;
                      const fillPrice = isSell && t.exit_price ? t.exit_price : t.entry_price;

                      return (
                        <tr
                          key={t._id}
                          onClick={() => setSelectedTrade(t)}
                          className="hover:bg-gray-50/80 dark:hover:bg-gray-700/40 transition-colors cursor-pointer group"
                        >
                          {/* Date */}
                          <td className="py-3.5 px-4 sm:px-6 font-mono text-xs text-gray-600 dark:text-gray-300">
                            {displayDate}
                          </td>

                          {/* Symbol */}
                          <td className="py-3.5 px-4">
                            <span className="font-bold text-gray-900 dark:text-white group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                              {t.symbol}
                            </span>
                          </td>

                          {/* Action Badge */}
                          <td className="py-3.5 px-4">
                            {t.trade_type === 'BUY' && (
                              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 dark:bg-emerald-950/80 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800">
                                BUY
                              </span>
                            )}
                            {t.trade_type === 'PYRAMID_ADD' && (
                              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-100 dark:bg-blue-950/80 text-blue-700 dark:text-blue-300 border border-blue-300 dark:border-blue-800">
                                PYRAMID +
                              </span>
                            )}
                            {t.trade_type === 'PARTIAL_SELL' && (
                              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-amber-100 dark:bg-amber-950/80 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-800">
                                TRIM (T1)
                              </span>
                            )}
                            {t.trade_type === 'SELL' && (
                              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-rose-100 dark:bg-rose-950/80 text-rose-700 dark:text-rose-300 border border-rose-300 dark:border-rose-800">
                                SELL
                              </span>
                            )}
                          </td>

                          {/* Fill Price */}
                          <td className="py-3.5 px-4 text-right font-medium font-mono text-gray-900 dark:text-gray-100">
                            ₹{Number(fillPrice).toFixed(2)}
                          </td>

                          {/* Qty & Value */}
                          <td className="py-3.5 px-4 text-right">
                            <div className="font-semibold text-gray-900 dark:text-gray-100">
                              {t.quantity} shares
                            </div>
                            <div className="text-[11px] text-gray-400">
                              ₹{Number(t.position_value || fillPrice * t.quantity).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                              {t.allocation_pct ? ` (${Number(t.allocation_pct).toFixed(0)}%)` : ''}
                            </div>
                          </td>

                          {/* Stop Loss & Target */}
                          <td className="py-3.5 px-4 text-right text-xs">
                            <div className="text-rose-500 dark:text-rose-400">
                              SL: ₹{t.stop_loss ? Number(t.stop_loss).toFixed(1) : '—'}
                            </div>
                            <div className="text-emerald-500 dark:text-emerald-400">
                              TP: ₹{t.target ? Number(t.target).toFixed(1) : '—'}
                            </div>
                          </td>

                          {/* P&L */}
                          <td className="py-3.5 px-4 text-right">
                            {hasPnl ? (
                              <div>
                                <span
                                  className={`font-semibold ${
                                    isProfit ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
                                  }`}
                                >
                                  {isProfit ? '+' : ''}₹{Number(pnl).toFixed(2)}
                                </span>
                                <div
                                  className={`text-[11px] font-medium ${
                                    isProfit ? 'text-emerald-500' : 'text-rose-500'
                                  }`}
                                >
                                  {isProfit ? '+' : ''}
                                  {Number(t.pnl_pct ?? 0).toFixed(2)}%
                                </div>
                              </div>
                            ) : (
                              <span className="text-gray-400 text-xs">—</span>
                            )}
                          </td>

                          {/* Pattern / Exit Reason */}
                          <td className="py-3.5 px-4 text-xs">
                            {t.exit_reason ? (
                              <span className="font-medium text-gray-800 dark:text-gray-200">
                                {t.exit_reason}
                              </span>
                            ) : t.entry_pattern ? (
                              <span className="text-gray-600 dark:text-gray-400 capitalize">
                                {t.entry_pattern.replace(/_/g, ' ')}
                              </span>
                            ) : (
                              <span className="text-gray-400">—</span>
                            )}
                          </td>

                          {/* Inspect Button */}
                          <td className="py-3.5 px-4 text-center">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedTrade(t);
                              }}
                              className="p-1.5 rounded-lg text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                            >
                              <ArrowsPointingOutIcon className="w-4 h-4" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}

            {/* Pagination Bar */}
            {totalPages > 1 && (
              <div className="p-4 border-t border-gray-200 dark:border-gray-700 flex flex-col sm:flex-row items-center justify-between gap-3 text-sm text-gray-600 dark:text-gray-300">
                <div>
                  Showing page <span className="font-semibold text-gray-900 dark:text-white">{page}</span> of{' '}
                  <span className="font-semibold text-gray-900 dark:text-white">{totalPages}</span> ({totalTrades} actions)
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page <= 1}
                    className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 disabled:opacity-40 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    <ChevronLeftIcon className="w-4 h-4" />
                    Previous
                  </button>

                  <button
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    disabled={page >= totalPages}
                    className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 disabled:opacity-40 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    Next
                    <ChevronRightIcon className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 2: BY SYMBOL BREAKDOWN */}
      {activeTab === 'symbols' && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
          <div className="p-5 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-base font-bold text-gray-900 dark:text-white">
              Stock Performance Breakdown
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Aggregated trading activity, total P&L, and win rates across all {symbolsSummary.length} traded stocks
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-50 dark:bg-gray-900 text-gray-600 dark:text-gray-300 uppercase text-xs tracking-wider border-b border-gray-200 dark:border-gray-700 font-semibold">
                <tr>
                  <th
                    onClick={() => handleSymSort('symbol')}
                    className="py-3.5 px-4 sm:px-6 cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Symbol"
                  >
                    <div className="flex items-center gap-1.5">
                      <span>Stock Symbol</span>
                      {renderSymSortIcon('symbol')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSymSort('total_pnl')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Total P&L"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Total P&L</span>
                      {renderSymSortIcon('total_pnl')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSymSort('win_rate')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Win Rate"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Win Rate</span>
                      {renderSymSortIcon('win_rate')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSymSort('winning_exits')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Wins / Losses"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Wins / Losses</span>
                      {renderSymSortIcon('winning_exits')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSymSort('buys')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Buys count"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Buys</span>
                      {renderSymSortIcon('buys')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSymSort('pyramids')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Pyramids count"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Pyramids</span>
                      {renderSymSortIcon('pyramids')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSymSort('sells')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Sells count"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Sells</span>
                      {renderSymSortIcon('sells')}
                    </div>
                  </th>
                  <th
                    onClick={() => handleSymSort('total_events')}
                    className="py-3.5 px-4 text-right cursor-pointer select-none group hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors"
                    title="Click to sort by Total Events"
                  >
                    <div className="flex items-center justify-end gap-1.5">
                      <span>Total Events</span>
                      {renderSymSortIcon('total_events')}
                    </div>
                  </th>
                  <th className="py-3.5 px-4 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700/80">
                {sortedSymbols.map((s) => {
                  const isProfit = s.total_pnl > 0;
                  return (
                    <tr
                      key={s.symbol}
                      className="hover:bg-gray-50/80 dark:hover:bg-gray-700/40 transition-colors"
                    >
                      <td className="py-3.5 px-4 sm:px-6 font-bold text-gray-900 dark:text-white">
                        {s.symbol}
                      </td>

                      <td className="py-3.5 px-4 text-right font-semibold">
                        <span
                          className={`${
                            isProfit ? 'text-emerald-600 dark:text-emerald-400' : s.total_pnl < 0 ? 'text-rose-600 dark:text-rose-400' : 'text-gray-400'
                          }`}
                        >
                          {isProfit ? '+' : ''}₹{s.total_pnl.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                        </span>
                      </td>

                      <td className="py-3.5 px-4 text-right font-medium text-gray-900 dark:text-gray-100">
                        {s.win_rate}%
                      </td>

                      <td className="py-3.5 px-4 text-right text-xs">
                        <span className="text-emerald-600 font-medium">{s.winning_exits}W</span>
                        {' / '}
                        <span className="text-rose-600 font-medium">{s.losing_exits}L</span>
                      </td>

                      <td className="py-3.5 px-4 text-right text-gray-600 dark:text-gray-300">
                        {s.buys}
                      </td>

                      <td className="py-3.5 px-4 text-right text-blue-600 dark:text-blue-400 font-medium">
                        {s.pyramids}
                      </td>

                      <td className="py-3.5 px-4 text-right text-gray-600 dark:text-gray-300">
                        {s.sells}
                      </td>

                      <td className="py-3.5 px-4 text-right font-bold text-gray-900 dark:text-gray-100">
                        {s.total_events}
                      </td>

                      <td className="py-3.5 px-4 text-center">
                        <button
                          onClick={() => {
                            setSearch(s.symbol);
                            setActiveTab('journal');
                          }}
                          className="px-2.5 py-1 rounded-md text-xs font-medium bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 hover:bg-indigo-100 transition-colors"
                        >
                          Filter Journal
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 2: YEAR-BY-YEAR PERFORMANCE BREAKDOWN */}
      {activeTab === 'yearly' && (
        <YearlyBreakdownTable
          breakdown={yearlyBreakdown}
          title={`Year-by-Year Performance Breakdown (${
            yearlyBreakdown.length > 5 ? '10-Year' : `${yearlyBreakdown.length}-Year`
          } Portfolio)`}
          subtitle="Annual compounded returns, Indian market context, and regime behavior for this simulation run"
          onFilterYear={(year) => {
            setSearch(year);
            setActiveTab('journal');
            setPage(1);
          }}
        />
      )}

      {/* TAB 4: STRATEGY CONFIGURATION */}
      {activeTab === 'config' && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 space-y-6">
          <div>
            <h2 className="text-lg font-bold text-gray-900 dark:text-white">
              Strategy & Simulation Parameters
            </h2>
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
              Exact configuration snapshot used during this backtest run
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Capital & Portfolio Config */}
            <div className="p-4 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/50 space-y-3">
              <h3 className="font-semibold text-sm text-gray-900 dark:text-white flex items-center gap-2">
                <CurrencyRupeeIcon className="w-4 h-4 text-indigo-500" />
                Capital & Execution Settings
              </h3>
              <div className="divide-y divide-gray-200 dark:divide-gray-700 text-xs">
                <div className="py-2 flex justify-between">
                  <span className="text-gray-500">Initial Capital</span>
                  <span className="font-mono font-medium text-gray-900 dark:text-white">
                    ₹{session?.capital_config?.initial_capital?.toLocaleString('en-IN') || '100,000'}
                  </span>
                </div>
                <div className="py-2 flex justify-between">
                  <span className="text-gray-500">Brokerage Charges</span>
                  <span className="font-mono font-medium text-gray-900 dark:text-white">
                    {(Number(session?.capital_config?.brokerage_charges || 0.0003) * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="py-2 flex justify-between">
                  <span className="text-gray-500">Slippage Tolerance</span>
                  <span className="font-mono font-medium text-gray-900 dark:text-white">
                    {(Number(session?.capital_config?.slippage_pct || 0.0015) * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="py-2 flex justify-between">
                  <span className="text-gray-500">Ranking Method</span>
                  <span className="font-mono font-medium text-gray-900 dark:text-white">
                    {session?.capital_config?.ranking_method || 'combined_score'}
                  </span>
                </div>
              </div>
            </div>

            {/* Strategy Snapshot */}
            <div className="p-4 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-900/50 space-y-3">
              <h3 className="font-semibold text-sm text-gray-900 dark:text-white flex items-center gap-2">
                <ShieldCheckIcon className="w-4 h-4 text-emerald-500" />
                Strategy Gates & Rules
              </h3>
              <div className="divide-y divide-gray-200 dark:divide-gray-700 text-xs">
                <div className="py-2 flex justify-between">
                  <span className="text-gray-500">Strategy Name</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {session?.strategy_name}
                  </span>
                </div>
                <div className="py-2 flex justify-between">
                  <span className="text-gray-500">Market Regime Gate</span>
                  <span className="font-medium text-emerald-600 dark:text-emerald-400">
                    200 SMA Active
                  </span>
                </div>
                <div className="py-2 flex justify-between">
                  <span className="text-gray-500">Min Market Cap Gate</span>
                  <span className="font-mono font-medium text-gray-900 dark:text-white">
                    ≥ ₹1,500 Cr
                  </span>
                </div>
                <div className="py-2 flex justify-between">
                  <span className="text-gray-500">Max Universe Size</span>
                  <span className="font-mono font-medium text-gray-900 dark:text-white">
                    {session?.total_symbols || 0} Symbols
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DETAILED TRADE INSPECTOR MODAL */}
      {selectedTrade && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-in fade-in duration-150">
          <div
            className="bg-white dark:bg-gray-900 rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-gray-200 dark:border-gray-700 space-y-5"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="flex items-start justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                    {selectedTrade.symbol}
                  </h3>
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-semibold ${
                      selectedTrade.trade_type === 'BUY'
                        ? 'bg-emerald-100 dark:bg-emerald-950 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800'
                        : selectedTrade.trade_type === 'PYRAMID_ADD'
                        ? 'bg-blue-100 dark:bg-blue-950 text-blue-700 dark:text-blue-300 border border-blue-300 dark:border-blue-800'
                        : selectedTrade.trade_type === 'PARTIAL_SELL'
                        ? 'bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-300 border border-amber-300 dark:border-amber-800'
                        : 'bg-rose-100 dark:bg-rose-950 text-rose-700 dark:text-rose-300 border border-rose-300 dark:border-rose-800'
                    }`}
                  >
                    {selectedTrade.trade_type}
                  </span>
                </div>
                <p className="text-xs text-gray-400 mt-1">
                  Action Date: {selectedTrade.exit_date || selectedTrade.entry_date}
                </p>
              </div>

              <button
                onClick={() => setSelectedTrade(null)}
                className="p-1 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>

            {/* Execution Details Grid */}
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-800/90 border border-gray-200 dark:border-gray-700">
                <span className="text-gray-400 block">Entry Date & Price</span>
                <span className="font-semibold text-gray-900 dark:text-white text-sm">
                  ₹{Number(selectedTrade.entry_price).toFixed(2)}
                </span>
                <div className="text-[11px] text-gray-500 mt-0.5">{selectedTrade.entry_date}</div>
              </div>

              <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-800/90 border border-gray-200 dark:border-gray-700">
                <span className="text-gray-400 block">Exit Date & Price</span>
                <span className="font-semibold text-gray-900 dark:text-white text-sm">
                  {selectedTrade.exit_price ? `₹${Number(selectedTrade.exit_price).toFixed(2)}` : '—'}
                </span>
                <div className="text-[11px] text-gray-500 mt-0.5">
                  {selectedTrade.exit_date || 'Open Position / Buy Event'}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-800/90 border border-gray-200 dark:border-gray-700">
                <span className="text-gray-400 block">Quantity & Value</span>
                <span className="font-semibold text-gray-900 dark:text-white text-sm">
                  {selectedTrade.quantity} Shares
                </span>
                <div className="text-[11px] text-gray-500 mt-0.5">
                  ₹{Number(selectedTrade.position_value || selectedTrade.entry_price * selectedTrade.quantity).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  {selectedTrade.allocation_pct ? ` (${Number(selectedTrade.allocation_pct).toFixed(1)}% of portfolio)` : ''}
                </div>
              </div>

              <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-800/90 border border-gray-200 dark:border-gray-700">
                <span className="text-gray-400 block">Net Realized P&L</span>
                <span
                  className={`font-semibold text-sm ${
                    (selectedTrade.pnl ?? 0) > 0
                      ? 'text-emerald-500'
                      : (selectedTrade.pnl ?? 0) < 0
                      ? 'text-rose-500'
                      : 'text-gray-400'
                  }`}
                >
                  {(selectedTrade.pnl ?? 0) > 0 ? '+' : ''}₹{Number(selectedTrade.pnl ?? 0).toFixed(2)}
                  {selectedTrade.pnl_pct ? ` (${Number(selectedTrade.pnl_pct).toFixed(2)}%)` : ''}
                </span>
                <div className="text-[11px] text-gray-500 mt-0.5">
                  {selectedTrade.exit_reason || 'In-flight'}
                </div>
              </div>
            </div>

            {/* Risk & Rules Snapshot */}
            <div className="p-4 rounded-xl bg-gray-50 dark:bg-gray-800/90 border border-gray-200 dark:border-gray-700 space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">Stop Loss Level</span>
                <span className="font-mono text-rose-500 font-semibold">
                  ₹{selectedTrade.stop_loss ? Number(selectedTrade.stop_loss).toFixed(2) : '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Profit Target</span>
                <span className="font-mono text-emerald-500 font-semibold">
                  ₹{selectedTrade.target ? Number(selectedTrade.target).toFixed(2) : '—'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Entry Pattern Trigger</span>
                <span className="font-medium text-gray-800 dark:text-gray-200 capitalize">
                  {selectedTrade.entry_pattern?.replace(/_/g, ' ') || 'Trend breakout'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Exit Reason</span>
                <span className="font-medium text-indigo-600 dark:text-indigo-400">
                  {selectedTrade.exit_reason || 'Active Trade'}
                </span>
              </div>
              {selectedTrade.cash_balance_at_entry !== undefined && (
                <div className="flex justify-between border-t border-gray-200 dark:border-gray-700 pt-2">
                  <span className="text-gray-400">Cash Balance at Entry</span>
                  <span className="font-mono text-gray-700 dark:text-gray-300">
                    ₹{Number(selectedTrade.cash_balance_at_entry).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </span>
                </div>
              )}
            </div>

            {/* Footer */}
            <div className="flex justify-end">
              <button
                onClick={() => setSelectedTrade(null)}
                className="px-4 py-2 rounded-xl bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200 font-medium text-xs hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors border border-transparent dark:border-gray-700"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function BacktestDetailPage() {
  return (
    <Suspense
      fallback={
        <div className="p-12 text-center text-gray-500 dark:text-gray-400">
          <div className="inline-block w-8 h-8 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mb-3"></div>
          <p className="text-sm font-semibold">Loading backtest simulation...</p>
        </div>
      }
    >
      <BacktestDetailPageContent />
    </Suspense>
  );
}

