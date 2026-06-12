'use client';

import { useState, useEffect, useMemo, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import {
  ArrowPathIcon,
  ArrowTrendingUpIcon,
  ShieldCheckIcon,
  BanknotesIcon,
  FlagIcon,
  XCircleIcon,
  PlusCircleIcon,
  PencilSquareIcon,
  ClockIcon,
  ChevronDownIcon,
} from '@heroicons/react/24/outline';
import { getPositions, getActivityLogs, Position } from '@/lib/api';

type LogAction =
  | 'POSITION_OPENED'
  | 'TRAIL_SL'
  | 'PYRAMID'
  | 'PARTIAL_SELL'
  | 'TARGET_HIT'
  | 'ENTRY_CORRECTION'
  | 'CLOSED';

interface ActivityLog {
  _id: string;
  symbol: string;
  action: LogAction;
  timestamp: string;
  details?: Record<string, any>;
}

const ACTION_CONFIG: Record<
  string,
  { label: string; icon: any; color: string; bgLight: string; bgDark: string; textLight: string; textDark: string }
> = {
  POSITION_OPENED: {
    label: 'Position Opened',
    icon: ArrowTrendingUpIcon,
    color: 'emerald',
    bgLight: 'bg-emerald-50',
    bgDark: 'dark:bg-emerald-900/20',
    textLight: 'text-emerald-700',
    textDark: 'dark:text-emerald-400',
  },
  TRAIL_SL: {
    label: 'Stop Loss Trailed',
    icon: ShieldCheckIcon,
    color: 'amber',
    bgLight: 'bg-amber-50',
    bgDark: 'dark:bg-amber-900/20',
    textLight: 'text-amber-700',
    textDark: 'dark:text-amber-400',
  },
  PYRAMID: {
    label: 'Pyramid Added',
    icon: PlusCircleIcon,
    color: 'blue',
    bgLight: 'bg-blue-50',
    bgDark: 'dark:bg-blue-900/20',
    textLight: 'text-blue-700',
    textDark: 'dark:text-blue-400',
  },
  PARTIAL_SELL: {
    label: 'Partial Exit',
    icon: BanknotesIcon,
    color: 'violet',
    bgLight: 'bg-violet-50',
    bgDark: 'dark:bg-violet-900/20',
    textLight: 'text-violet-700',
    textDark: 'dark:text-violet-400',
  },
  TARGET_HIT: {
    label: 'Target Hit',
    icon: FlagIcon,
    color: 'green',
    bgLight: 'bg-green-50',
    bgDark: 'dark:bg-green-900/20',
    textLight: 'text-green-700',
    textDark: 'dark:text-green-400',
  },
  ENTRY_CORRECTION: {
    label: 'Entry Corrected',
    icon: PencilSquareIcon,
    color: 'yellow',
    bgLight: 'bg-yellow-50',
    bgDark: 'dark:bg-yellow-900/20',
    textLight: 'text-yellow-700',
    textDark: 'dark:text-yellow-400',
  },
  CLOSED: {
    label: 'Position Closed',
    icon: XCircleIcon,
    color: 'red',
    bgLight: 'bg-red-50',
    bgDark: 'dark:bg-red-900/20',
    textLight: 'text-red-700',
    textDark: 'dark:text-red-400',
  },
};

const formatDateTime = (d: string) => {
  if (!d) return '—';
  try {
    return new Date(d).toLocaleString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true,
    });
  } catch {
    return d;
  }
};

const formatDate = (d: string) => {
  if (!d) return '—';
  try {
    return new Date(d).toLocaleDateString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
    });
  } catch {
    return d;
  }
};

const formatTime = (d: string) => {
  if (!d) return '';
  try {
    return new Date(d).toLocaleTimeString('en-IN', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true,
    });
  } catch {
    return '';
  }
};

export default function ActivityLogPage() {
  return (
    <Suspense
      fallback={
        <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="animate-pulse space-y-8">
            <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded-lg w-64" />
            <div className="h-48 bg-gray-200 dark:bg-gray-700 rounded-2xl" />
            <div className="h-96 bg-gray-200 dark:bg-gray-700 rounded-2xl" />
          </div>
        </div>
      }
    >
      <ActivityLogContent />
    </Suspense>
  );
}

function ActivityLogContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const urlSymbol = searchParams.get('symbol') || '';

  const [allPositions, setAllPositions] = useState<Position[]>([]);
  const [allLogs, setAllLogs] = useState<ActivityLog[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>(urlSymbol);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    const [posRes, logsRes] = await Promise.all([
      getPositions(),
      getActivityLogs(undefined, 500),
    ]);
    if (posRes.status === 'success') setAllPositions(posRes.positions);
    if (logsRes.status === 'success') setAllLogs(logsRes.logs as ActivityLog[]);
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    if (urlSymbol && urlSymbol !== selectedSymbol) {
      setSelectedSymbol(urlSymbol);
    }
  }, [urlSymbol]);

  const uniqueSymbols = useMemo(() => {
    const symbolSet = new Set<string>();
    allPositions.forEach((p) => symbolSet.add(p.symbol));
    allLogs.forEach((l) => symbolSet.add(l.symbol));
    return Array.from(symbolSet).sort();
  }, [allPositions, allLogs]);

  const symbolPositions = useMemo(() => {
    return allPositions.filter((p) => p.symbol === selectedSymbol);
  }, [allPositions, selectedSymbol]);

  const symbolLogs = useMemo(() => {
    return allLogs
      .filter((l) => l.symbol === selectedSymbol)
      .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  }, [allLogs, selectedSymbol]);

  const timeline = useMemo(() => {
    if (!selectedSymbol) return [];

    const events: Array<{
      id: string;
      type: string;
      timestamp: string;
      data: Record<string, any>;
    }> = [];

    symbolPositions.forEach((pos) => {
      events.push({
        id: `open-${pos._id}`,
        type: 'POSITION_OPENED',
        timestamp: pos.entry_date || pos.created_at,
        data: {
          entry_price: pos.entry_price,
          quantity: pos.initial_quantity || pos.quantity,
          stop_loss: pos.stop_loss,
          target: pos.target,
          total_investment: pos.total_investment,
          strategy: pos.strategy_name,
        },
      });

      pos.updates?.forEach((u: any, idx: number) => {
        events.push({
          id: `update-${pos._id}-${idx}`,
          type: u.type || 'UNKNOWN',
          timestamp: u.date || u.created_at || pos.created_at,
          data: u,
        });
      });

      if (pos.status === 'CLOSED') {
        events.push({
          id: `close-${pos._id}`,
          type: 'CLOSED',
          timestamp: pos.exit_date || pos.updated_at,
          data: {
            exit_price: pos.exit_price,
            exit_reason: pos.exit_reason,
            pnl_pct: pos.pnl_pct,
          },
        });
      }
    });

    symbolLogs.forEach((log) => {
      const existing = events.find(
        (e) =>
          e.type === log.action &&
          Math.abs(new Date(e.timestamp).getTime() - new Date(log.timestamp).getTime()) < 5000
      );
      if (!existing) {
        events.push({
          id: `log-${log._id}`,
          type: log.action,
          timestamp: log.timestamp,
          data: log.details || {},
        });
      }
    });

    return events.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
  }, [selectedSymbol, symbolPositions, symbolLogs]);

  const currentPos = symbolPositions.find((p) => p.status === 'OPEN') || symbolPositions[0];

  if (loading) {
    return (
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <div className="h-8 w-64 bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse" />
            <div className="h-4 w-80 bg-gray-200 dark:bg-gray-700 rounded mt-2 animate-pulse" />
          </div>
          <div className="h-10 w-28 bg-gray-200 dark:bg-gray-700 rounded-lg animate-pulse" />
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
          <div className="h-3 w-24 bg-gray-200 dark:bg-gray-700 rounded animate-pulse mb-3" />
          <div className="h-12 w-full max-w-md bg-gray-200 dark:bg-gray-700 rounded-xl animate-pulse" />
        </div>

        {selectedSymbol && (
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
            <div className="px-6 py-5 border-b border-gray-100 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  <div className="h-6 w-32 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                  <div className="h-4 w-48 bg-gray-200 dark:bg-gray-700 rounded mt-2 animate-pulse" />
                </div>
                <div className="flex gap-4">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="text-center">
                      <div className="h-3 w-10 bg-gray-200 dark:bg-gray-700 rounded animate-pulse mb-1" />
                      <div className="h-5 w-16 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="relative">
                <div className="absolute left-5 top-5 bottom-5 w-px bg-gray-200 dark:bg-gray-700" />
                <div className="space-y-6">
                  {[...Array(4)].map((_, i) => (
                    <div key={i} className="relative flex gap-5 pb-6 last:pb-0">
                      <div className="relative z-10 flex-shrink-0 w-10 h-10 rounded-full bg-gray-200 dark:bg-gray-700 animate-pulse border-2 border-white dark:border-gray-800" />
                      <div className="flex-1">
                        <div className="flex items-baseline justify-between gap-3 mb-1.5">
                          <div className="h-4 w-32 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                          <div className="h-3 w-24 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                        </div>
                        <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-3.5 border border-gray-100 dark:border-gray-700/50 space-y-2">
                          <div className="flex justify-between">
                            <div className="h-3 w-20 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                            <div className="h-3 w-16 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                          </div>
                          <div className="flex justify-between">
                            <div className="h-3 w-24 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                            <div className="h-3 w-16 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                          </div>
                          <div className="flex justify-between">
                            <div className="h-3 w-16 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                            <div className="h-3 w-16 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
            Activity Timeline
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Track position lifecycle — from entry to exit
          </p>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className="inline-flex items-center gap-2 px-4 py-2.5 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition disabled:opacity-50"
        >
          <ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Symbol Selector */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <label className="block text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-3">
          Select Stock
        </label>
        <div className="relative max-w-md">
          <select
            value={selectedSymbol}
            onChange={(e) => {
              setSelectedSymbol(e.target.value);
              router.replace('/activity-log', { scroll: false });
            }}
            disabled={loading}
            className="w-full px-4 py-3 pr-10 border border-gray-200 dark:border-gray-600 rounded-xl bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 text-base font-medium appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50 transition"
          >
            <option value="">
              {loading
                ? 'Loading...'
                : uniqueSymbols.length === 0
                ? 'No stocks found'
                : 'Choose a stock...'}
            </option>
            {uniqueSymbols.map((sym) => {
              const openPos = allPositions.find((p) => p.symbol === sym && p.status === 'OPEN');
              const closedPos = allPositions.filter((p) => p.symbol === sym && p.status === 'CLOSED').length;
              let label = sym;
              if (openPos) label += ' — Open';
              else if (closedPos > 0) label += ` — Closed (${closedPos})`;
              return (
                <option key={sym} value={sym}>
                  {label}
                </option>
              );
            })}
          </select>
          <ChevronDownIcon className="absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400 pointer-events-none" />
        </div>
      </div>

      {/* Timeline */}
      {selectedSymbol ? (
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
          {/* Position Summary Header */}
          {currentPos && (
            <div className="px-6 py-5 border-b border-gray-100 dark:border-gray-700 bg-gradient-to-r from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div>
                  <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
                    {selectedSymbol}
                  </h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                    {currentPos.strategy_name} · Entry {formatDate(currentPos.entry_date || currentPos.created_at)}
                  </p>
                </div>
                <div className="flex items-center gap-4 text-sm">
                  <div className="text-center">
                    <p className="text-xs text-gray-500 dark:text-gray-400">Entry</p>
                    <p className="font-semibold text-gray-900 dark:text-gray-100">
                      ₹{currentPos.entry_price?.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-gray-500 dark:text-gray-400">Qty</p>
                    <p className="font-semibold text-gray-900 dark:text-gray-100">
                      {currentPos.quantity}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-gray-500 dark:text-gray-400">SL</p>
                    <p className="font-semibold text-red-600 dark:text-red-400">
                      ₹{(currentPos.current_stop_loss || currentPos.stop_loss)?.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-gray-500 dark:text-gray-400">Target</p>
                    <p className="font-semibold text-green-600 dark:text-green-400">
                      ₹{(currentPos.current_target || currentPos.target)?.toFixed(2)}
                    </p>
                  </div>
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-semibold ${
                      currentPos.status === 'OPEN'
                        ? 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400'
                        : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    {currentPos.status}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Timeline Events */}
          <div className="p-6">
            {timeline.length === 0 ? (
              <div className="text-center py-12">
                <ClockIcon className="h-12 w-12 mx-auto text-gray-300 dark:text-gray-600 mb-3" />
                <p className="text-gray-500 dark:text-gray-400 text-sm">
                  No activity recorded for {selectedSymbol} yet.
                </p>
              </div>
            ) : (
              <div className="relative">
                {/* Vertical line */}
                <div className="absolute left-5 top-5 bottom-5 w-px bg-gray-200 dark:bg-gray-700" />

                <div className="space-y-0">
                  {timeline.map((event, idx) => {
                    const config = ACTION_CONFIG[event.type] || {
                      label: event.type,
                      icon: ClockIcon,
                      color: 'gray',
                      bgLight: 'bg-gray-50',
                      bgDark: 'dark:bg-gray-700',
                      textLight: 'text-gray-700',
                      textDark: 'dark:text-gray-300',
                    };
                    const Icon = config.icon;

                    return (
                      <div key={event.id} className="relative flex gap-5 pb-6 last:pb-0">
                        {/* Icon */}
                        <div
                          className={`relative z-10 flex-shrink-0 w-10 h-10 rounded-full ${config.bgLight} ${config.bgDark} flex items-center justify-center border-2 border-white dark:border-gray-800`}
                        >
                          <Icon className={`h-5 w-5 ${config.textLight} ${config.textDark}`} />
                        </div>

                        {/* Content */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-baseline justify-between gap-3 mb-1.5">
                            <h3
                              className={`text-sm font-semibold ${config.textLight} ${config.textDark}`}
                            >
                              {config.label}
                            </h3>
                            <span className="text-xs text-gray-400 dark:text-gray-500 whitespace-nowrap">
                              {formatDateTime(event.timestamp)}
                            </span>
                          </div>

                          {/* Event Details */}
                          <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-3.5 text-sm space-y-1 border border-gray-100 dark:border-gray-700/50">
                            {event.type === 'POSITION_OPENED' && (
                              <>
                                <DetailRow label="Entry Price" value={`₹${event.data.entry_price?.toFixed(2)}`} />
                                <DetailRow label="Quantity" value={String(event.data.quantity)} />
                                <DetailRow
                                  label="Stop Loss"
                                  value={`₹${event.data.stop_loss?.toFixed(2)}`}
                                  valueClass="text-red-600 dark:text-red-400"
                                />
                                <DetailRow
                                  label="Target"
                                  value={`₹${event.data.target?.toFixed(2)}`}
                                  valueClass="text-green-600 dark:text-green-400"
                                />
                                <DetailRow
                                  label="Investment"
                                  value={`₹${event.data.total_investment?.toLocaleString('en-IN')}`}
                                />
                                {event.data.strategy && (
                                  <DetailRow label="Strategy" value={event.data.strategy} />
                                )}
                              </>
                            )}
                            {event.type === 'TRAIL_SL' && (
                              <>
                                {event.data.prev_sl != null && (
                                  <DetailRow label="Previous SL" value={`₹${event.data.prev_sl?.toFixed(2)}`} />
                                )}
                                {event.data.current_sl != null && (
                                  <DetailRow
                                    label="New SL"
                                    value={`₹${event.data.current_sl?.toFixed(2)}`}
                                    valueClass="font-semibold text-amber-600 dark:text-amber-400"
                                  />
                                )}
                                {event.data.prev_sl != null && event.data.current_sl != null && (
                                  <DetailRow
                                    label="Change"
                                    value={`+₹${(event.data.current_sl - event.data.prev_sl).toFixed(2)}`}
                                    valueClass="text-emerald-600 dark:text-emerald-400"
                                  />
                                )}
                              </>
                            )}
                            {event.type === 'PYRAMID' && (
                              <>
                                {event.data.entry_price != null && (
                                  <DetailRow label="Pyramid Price" value={`₹${event.data.entry_price?.toFixed(2)}`} />
                                )}
                                {event.data.quantity != null && (
                                  <DetailRow label="Added Qty" value={String(event.data.quantity)} />
                                )}
                                {event.data.adds_count != null && (
                                  <DetailRow label="Total Adds" value={String(event.data.adds_count)} />
                                )}
                              </>
                            )}
                            {event.type === 'PARTIAL_SELL' && (
                              <>
                                {event.data.exit_price != null && (
                                  <DetailRow label="Sell Price" value={`₹${event.data.exit_price?.toFixed(2)}`} />
                                )}
                                {event.data.quantity != null && (
                                  <DetailRow label="Sold Qty" value={String(event.data.quantity)} />
                                )}
                                {event.data.pnl_pct != null && (
                                  <DetailRow
                                    label="PnL"
                                    value={`${event.data.pnl_pct?.toFixed(2)}%`}
                                    valueClass={event.data.pnl_pct >= 0 ? 'text-green-600' : 'text-red-600'}
                                  />
                                )}
                              </>
                            )}
                            {event.type === 'TARGET_HIT' && (
                              <>
                                {event.data.target_price != null && (
                                  <DetailRow label="Target Price" value={`₹${event.data.target_price?.toFixed(2)}`} />
                                )}
                                {event.data.target_name && (
                                  <DetailRow label="Target" value={event.data.target_name} />
                                )}
                              </>
                            )}
                            {event.type === 'ENTRY_CORRECTION' && (
                              <>
                                {event.data.old_entry != null && (
                                  <DetailRow label="Old Entry" value={`₹${event.data.old_entry?.toFixed(2)}`} />
                                )}
                                {event.data.entry_price != null && (
                                  <DetailRow
                                    label="New Entry"
                                    value={`₹${event.data.entry_price?.toFixed(2)}`}
                                    valueClass="font-semibold"
                                  />
                                )}
                                {event.data.reason && <DetailRow label="Reason" value={event.data.reason} />}
                              </>
                            )}
                            {event.type === 'CLOSED' && (
                              <>
                                {event.data.exit_price != null && (
                                  <DetailRow label="Exit Price" value={`₹${event.data.exit_price?.toFixed(2)}`} />
                                )}
                                {event.data.exit_reason && (
                                  <DetailRow label="Reason" value={event.data.exit_reason} />
                                )}
                                {event.data.pnl_pct != null && (
                                  <DetailRow
                                    label="Final PnL"
                                    value={`${event.data.pnl_pct?.toFixed(2)}%`}
                                    valueClass={`text-base font-bold ${
                                      event.data.pnl_pct >= 0 ? 'text-green-600' : 'text-red-600'
                                    }`}
                                  />
                                )}
                              </>
                            )}
                            {/* Fallback for unknown types */}
                            {!['POSITION_OPENED', 'TRAIL_SL', 'PYRAMID', 'PARTIAL_SELL', 'TARGET_HIT', 'ENTRY_CORRECTION', 'CLOSED'].includes(event.type) && (
                              <pre className="text-xs text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
                                {JSON.stringify(event.data, null, 2)}
                              </pre>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-16 text-center">
          <ClockIcon className="h-16 w-16 mx-auto text-gray-200 dark:text-gray-700 mb-4" />
          <p className="text-gray-500 dark:text-gray-400 text-base font-medium">
            Select a stock to view its complete activity timeline
          </p>
          <p className="text-gray-400 dark:text-gray-500 text-sm mt-1">
            Includes entries, trailing stops, pyramiding, and exits
          </p>
        </div>
      )}
    </div>
  );
}

function DetailRow({
  label,
  value,
  valueClass = 'text-gray-900 dark:text-gray-100',
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-gray-500 dark:text-gray-400 text-xs">{label}</span>
      <span className={`font-medium text-xs ${valueClass}`}>{value}</span>
    </div>
  );
}
