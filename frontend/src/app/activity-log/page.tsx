'use client';

import { useState, useEffect } from 'react';
import { ChevronDownIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import { getPositions, getActivityLogs, Position } from '@/lib/api';

export default function ActivityLogPage() {
  const [allPositions, setAllPositions] = useState<Position[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [globalLogs, setGlobalLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    const [posRes, logsRes] = await Promise.all([getPositions(), getActivityLogs()]);
    if (posRes.status === 'success') setAllPositions(posRes.positions);
    if (logsRes.status === 'success') setGlobalLogs(logsRes.logs);
    setLoading(false);
  };

  useEffect(() => { fetchData(); }, []);

  const selectedPos = allPositions.find(p => p._id === selectedId);

  const formatDate = (d: string) => {
    if (!d) return '—';
    try {
      return new Date(d).toLocaleString('en-IN', {
        day: '2-digit', month: 'short', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    } catch {
      return d;
    }
  };

  const getUpdateStyle = (type: string) => {
    switch (type) {
      case 'POSITION_OPENED':
      case 'PYRAMID': return { bg: 'bg-blue-100 dark:bg-blue-900/30', text: 'text-blue-700 dark:text-blue-300', label: type === 'POSITION_OPENED' ? 'Position Opened' : 'Pyramid Buy' };
      case 'TRAIL_SL': return { bg: 'bg-orange-100 dark:bg-orange-900/30', text: 'text-orange-700 dark:text-orange-300', label: 'SL Trail' };
      case 'PARTIAL_SELL': return { bg: 'bg-purple-100 dark:bg-purple-900/30', text: 'text-purple-700 dark:text-purple-300', label: 'Partial Sell' };
      case 'TARGET_HIT': return { bg: 'bg-green-100 dark:bg-green-900/30', text: 'text-green-700 dark:text-green-300', label: 'Target Hit' };
      case 'ENTRY_CORRECTION': return { bg: 'bg-yellow-100 dark:bg-yellow-900/30', text: 'text-yellow-700 dark:text-yellow-300', label: 'Entry Correction' };
      case 'CLOSED': return { bg: 'bg-red-100 dark:bg-red-900/30', text: 'text-red-700 dark:text-red-300', label: 'Closed' };
      default: return { bg: 'bg-gray-100 dark:bg-gray-700', text: 'text-gray-700 dark:text-gray-300', label: type };
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Position Activity Log</h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">Global timeline and per-position update history</p>
        </div>
        <button onClick={fetchData} className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
          <ArrowPathIcon className="h-5 w-5" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Global Timeline */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-4">Global Activity Timeline</h2>
        {globalLogs.length === 0 ? (
          <p className="text-gray-500 dark:text-gray-400 text-sm">
            {loading ? 'Loading...' : 'No activity logs yet. Run a trading cycle to generate logs.'}
          </p>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {globalLogs.map((log: any, idx: number) => {
              const style = getUpdateStyle(log.action);
              return (
                <div key={log._id || idx} className={`flex items-center gap-3 px-4 py-3 rounded-lg ${style.bg}`}>
                  <span className={`px-2.5 py-1 rounded text-xs font-semibold ${style.text} bg-white/60 dark:bg-gray-800/60 whitespace-nowrap`}>
                    {style.label}
                  </span>
                  <span className="font-semibold text-gray-900 dark:text-gray-100 text-sm">{log.symbol}</span>
                  <span className="text-xs text-gray-500 dark:text-gray-400 ml-auto whitespace-nowrap">
                    {formatDate(log.timestamp)}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Position Selector */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Select Position (detailed view)</label>
        <div className="relative">
          <select
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
            disabled={loading}
            className="w-full md:w-96 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 appearance-none cursor-pointer disabled:opacity-50"
          >
            <option value="">
              {loading ? 'Loading positions...' : (allPositions.length === 0 ? 'No positions found' : 'Choose a position...')}
            </option>
            {allPositions.map((p) => (
              <option key={p._id} value={p._id}>
                {p.symbol} [{p.status}] — Entry: ₹{p.entry_price?.toFixed(2)} | Qty: {p.quantity} | {p.entry_date ? new Date(p.entry_date).toLocaleDateString('en-IN') : ''}
              </option>
            ))}
          </select>
          <ChevronDownIcon className="absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400 pointer-events-none" />
        </div>
      </div>

      {/* Per-Position Timeline */}
      {selectedPos ? (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">{selectedPos.symbol}</h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {selectedPos.status} — {selectedPos.strategy_name} — {selectedPos.quantity} qty @ ₹{selectedPos.entry_price?.toFixed(2)}
              </p>
            </div>
            <span className={`px-3 py-1.5 rounded-full text-sm font-semibold ${
              selectedPos.status === 'OPEN'
                ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
            }`}>
              {selectedPos.status}
            </span>
          </div>

          <div className="space-y-0">
            {/* Step 1: Initial Buy */}
            <div className="flex gap-4">
              <div className="flex flex-col items-center">
                <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center flex-shrink-0">
                  <span className="text-green-600 dark:text-green-400 text-sm font-bold">1</span>
                </div>
                <div className="w-0.5 flex-1 bg-gray-200 dark:bg-gray-700 my-1" />
              </div>
              <div className="pb-6 flex-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className="px-2.5 py-1 rounded text-xs font-semibold bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300">Initial Buy</span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">{formatDate(selectedPos.created_at)}</span>
                </div>
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 text-sm space-y-1.5">
                  <p className="text-gray-700 dark:text-gray-300">Entry: <span className="font-semibold text-gray-900 dark:text-gray-100">₹{selectedPos.entry_price?.toFixed(2)}</span></p>
                  <p className="text-gray-700 dark:text-gray-300">Quantity: <span className="font-semibold">{selectedPos.quantity}</span></p>
                  <p className="text-gray-700 dark:text-gray-300">Stop Loss: <span className="font-semibold text-red-600">₹{selectedPos.stop_loss?.toFixed(2)}</span></p>
                  <p className="text-gray-700 dark:text-gray-300">Target: <span className="font-semibold text-green-600">₹{selectedPos.target?.toFixed(2)}</span></p>
                  <p className="text-gray-700 dark:text-gray-300">Investment: <span className="font-semibold">₹{selectedPos.total_investment?.toLocaleString()}</span></p>
                  <p className="text-gray-700 dark:text-gray-300">Strategy: {selectedPos.strategy_name}</p>
                </div>
              </div>
            </div>

            {/* Updates */}
            {selectedPos.updates && selectedPos.updates.length > 0 ? (
              selectedPos.updates.map((update: any, idx: number) => {
                const style = getUpdateStyle(update.type);
                const isLast = idx === (selectedPos.updates?.length || 0) - 1 && selectedPos.status !== 'CLOSED';
                return (
                  <div key={idx} className="flex gap-4">
                    <div className="flex flex-col items-center">
                      <div className={`w-10 h-10 rounded-full ${style.bg} flex items-center justify-center flex-shrink-0`}>
                        <span className={`${style.text} text-sm font-bold`}>{idx + 2}</span>
                      </div>
                      {!isLast && <div className="w-0.5 flex-1 bg-gray-200 dark:bg-gray-700 my-1" />}
                    </div>
                    <div className="pb-6 flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`px-2.5 py-1 rounded text-xs font-semibold ${style.bg} ${style.text}`}>{style.label}</span>
                        <span className="text-xs text-gray-500 dark:text-gray-400">{formatDate(update.date)}</span>
                      </div>
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 text-sm space-y-1.5">
                        {update.entry_price && <p className="text-gray-700 dark:text-gray-300">Entry Price: ₹{update.entry_price?.toFixed(2)}</p>}
                        {update.prev_sl !== undefined && <p className="text-gray-700 dark:text-gray-300">Previous SL: ₹{update.prev_sl?.toFixed(2)}</p>}
                        {update.current_sl !== undefined && <p className="text-gray-700 dark:text-gray-300">New SL: <span className="font-semibold text-orange-600">₹{update.current_sl?.toFixed(2)}</span></p>}
                        {update.quantity !== undefined && update.type !== 'CLOSED' && <p className="text-gray-700 dark:text-gray-300">Quantity: {update.quantity}</p>}
                        {update.exit_price !== undefined && <p className="text-gray-700 dark:text-gray-300">Exit Price: ₹{update.exit_price?.toFixed(2)}</p>}
                        {update.exit_reason && <p className="text-gray-700 dark:text-gray-300">Reason: {update.exit_reason}</p>}
                        {update.pnl_pct !== undefined && (
                          <p className="text-gray-700 dark:text-gray-300">
                            PnL: <span className={`font-semibold ${update.pnl_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>{update.pnl_pct?.toFixed(2)}%</span>
                          </p>
                        )}
                        {update.reason && <p className="text-gray-700 dark:text-gray-300">Note: {update.reason}</p>}
                      </div>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="flex gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-10 h-10 rounded-full bg-gray-100 dark:bg-gray-700 flex items-center justify-center flex-shrink-0">
                    <span className="text-gray-400 text-xs">--</span>
                  </div>
                </div>
                <div className="pb-6">
                  <p className="text-gray-500 dark:text-gray-400 text-sm">No updates recorded yet for this position.</p>
                </div>
              </div>
            )}

            {/* Closed status */}
            {selectedPos.status === 'CLOSED' && (
              <div className="flex gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center flex-shrink-0">
                    <span className="text-red-600 dark:text-red-400 text-sm font-bold">X</span>
                  </div>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="px-2.5 py-1 rounded text-xs font-semibold bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300">Position Closed</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">{formatDate(selectedPos.exit_date || '')}</span>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 text-sm space-y-1.5">
                    <p className="text-gray-700 dark:text-gray-300">Exit Price: ₹{selectedPos.exit_price?.toFixed(2)}</p>
                    <p className="text-gray-700 dark:text-gray-300">Exit Reason: {selectedPos.exit_reason}</p>
                    <p className="text-gray-700 dark:text-gray-300">
                      Final PnL: <span className={`font-bold text-lg ${(selectedPos.pnl_pct || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>{selectedPos.pnl_pct?.toFixed(2)}%</span>
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-12 text-center">
          <p className="text-gray-500 dark:text-gray-400 text-lg">
            {allPositions.length === 0
              ? 'No positions found. Run a trading cycle to generate positions.'
              : 'Select a position from the dropdown above to view its full activity history.'}
          </p>
        </div>
      )}
    </div>
  );
}
