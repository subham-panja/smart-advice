'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  PencilIcon,
  XMarkIcon,
  TrashIcon,
  CheckIcon,
  ArrowPathIcon,
  EyeIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
} from '@heroicons/react/24/outline';
import {
  getPositions,
  updatePosition,
  closePosition,
  getCycleStats,
  Position,
  CycleStats,
} from '@/lib/api';

export default function PositionsPage() {
  const router = useRouter();
  const [positions, setPositions] = useState<Position[]>([]);
  const [stats, setStats] = useState<Partial<CycleStats>>({});
  const [loading, setLoading] = useState(true);
  const [editModalPos, setEditModalPos] = useState<Position | null>(null);
  const [entryPrice, setEntryPrice] = useState('');
  const [pyramidModalPos, setPyramidModalPos] = useState<Position | null>(null);
  const [pyramidEdits, setPyramidEdits] = useState<Array<{ qty: number; price: string }>>([]);
  const [actionMsg, setActionMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const fetchData = async () => {
    setLoading(true);
    const [posRes, statsRes] = await Promise.all([
      getPositions('OPEN'),
      getCycleStats(),
    ]);
    if (posRes.status === 'success') setPositions(posRes.positions);
    if (statsRes.status === 'success') setStats(statsRes);
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  const openEditModal = (pos: Position) => {
    setEditModalPos(pos);
    setEntryPrice(String(pos.entry_price || ''));
  };

  const handleSaveEntryPrice = async () => {
    if (!editModalPos || !entryPrice) return;
    const newPrice = parseFloat(entryPrice);
    if (isNaN(newPrice) || newPrice <= 0) {
      setActionMsg({ type: 'error', text: 'Invalid entry price' });
      return;
    }

    const res = await updatePosition(editModalPos.symbol, { entry_price: newPrice });
    if (res.status === 'success') {
      setActionMsg({ type: 'success', text: `${editModalPos.symbol} entry price updated to ₹${newPrice.toFixed(2)}` });
      setEditModalPos(null);
      setEntryPrice('');
      fetchData();
    } else {
      setActionMsg({ type: 'error', text: res.error || 'Update failed' });
    }
  };

  const handleClose = async (symbol: string) => {
    if (!confirm(`Close position for ${symbol}? This action cannot be undone.`)) return;
    const res = await closePosition(symbol);
    if (res.status === 'success') {
      setActionMsg({ type: 'success', text: `${symbol} position closed` });
      fetchData();
    } else {
      setActionMsg({ type: 'error', text: res.error || 'Close failed' });
    }
  };

  const viewTimeline = (symbol: string) => {
    router.push(`/activity-log?symbol=${encodeURIComponent(symbol)}`);
  };

  const openPyramidModal = (pos: Position) => {
    const pyramidUpdates = (pos.updates || []).filter((u: any) => u.type === 'PYRAMID');
    const edits = pyramidUpdates.map((u: any) => ({
      qty: u.quantity || 0,
      price: String(u.entry_price || ''),
    }));
    setPyramidModalPos(pos);
    setPyramidEdits(edits);
  };

  const handleSavePyramids = async () => {
    if (!pyramidModalPos) return;

    // Calculate original position investment (before pyramids)
    const initialQty = pyramidModalPos.initial_quantity || pyramidModalPos.quantity;
    const currentPyramids = (pyramidModalPos.updates || []).filter((u: any) => u.type === 'PYRAMID');
    
    // Get the original entry price (first buy)
    const openUpdate = (pyramidModalPos.updates || []).find((u: any) => u.type === 'POSITION_OPENED');
    const originalPrice = openUpdate?.entry_price || pyramidModalPos.entry_price;
    const originalQty = initialQty - currentPyramids.reduce((sum, u: any) => sum + (u.quantity || 0), 0);
    
    let totalInvestment = originalQty * originalPrice;
    
    // Add each pyramid's investment
    pyramidEdits.forEach((edit) => {
      const qty = edit.qty;
      const price = parseFloat(edit.price);
      if (!isNaN(price) && qty > 0) {
        totalInvestment += qty * price;
      }
    });

    // Calculate new average entry price
    const newAvgPrice = totalInvestment / pyramidModalPos.quantity;

    const res = await updatePosition(pyramidModalPos.symbol, {
      entry_price: parseFloat(newAvgPrice.toFixed(2)),
      total_investment: parseFloat(totalInvestment.toFixed(2)),
    });

    if (res.status === 'success') {
      setActionMsg({
        type: 'success',
        text: `${pyramidModalPos.symbol} pyramid prices updated. New avg: ₹${newAvgPrice.toFixed(2)}`,
      });
      setPyramidModalPos(null);
      setPyramidEdits([]);
      fetchData();
    } else {
      setActionMsg({ type: 'error', text: res.error || 'Update failed' });
    }
  };

  const totalInvested = positions.reduce((sum, p) => sum + (p.total_investment || 0), 0);
  const currentMarketValue = positions.reduce(
    (sum, p) => sum + (p.current_price || p.entry_price) * p.quantity,
    0
  );
  const unrealizedPnL = currentMarketValue - totalInvested;
  const unrealizedPnLPct = totalInvested > 0 ? (unrealizedPnL / totalInvested) * 100 : 0;

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
            Active Positions
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Manage and monitor your open positions
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

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Open Positions"
          value={String(positions.length)}
          subtitle={`of ${stats.initial_capital ? '₹' + stats.initial_capital?.toLocaleString('en-IN') : '—'} capital`}
        />
        <StatCard
          label="Total Invested"
          value={`₹${totalInvested.toLocaleString('en-IN')}`}
          subtitle={`${positions.length > 0 ? (totalInvested / positions.length).toLocaleString('en-IN', { maximumFractionDigits: 0 }) : 0} avg/position`}
        />
        <StatCard
          label="Unrealized P&L"
          value={`${unrealizedPnL >= 0 ? '+' : ''}₹${unrealizedPnL.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`}
          subtitle={`${unrealizedPnLPct >= 0 ? '+' : ''}${unrealizedPnLPct.toFixed(2)}%`}
          positive={unrealizedPnL >= 0}
          icon={unrealizedPnL >= 0 ? ArrowTrendingUpIcon : ArrowTrendingDownIcon}
        />
        <StatCard
          label="Cash Available"
          value={`₹${(stats.cash_remaining || 0).toLocaleString('en-IN')}`}
          subtitle={`${stats.initial_capital ? ((stats.cash_remaining || 0) / stats.initial_capital * 100).toFixed(1) : 0}% of capital`}
        />
      </div>

      {/* Action Message */}
      {actionMsg && (
        <div
          className={`flex items-center gap-3 p-4 rounded-xl border ${
            actionMsg.type === 'success'
              ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-200'
              : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-800 dark:text-red-200'
          }`}
        >
          {actionMsg.type === 'success' ? (
            <CheckIcon className="h-5 w-5 flex-shrink-0" />
          ) : (
            <XMarkIcon className="h-5 w-5 flex-shrink-0" />
          )}
          <span className="text-sm font-medium">{actionMsg.text}</span>
          <button
            onClick={() => setActionMsg(null)}
            className="ml-auto p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 transition"
          >
            <XMarkIcon className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Positions Table */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 dark:border-gray-700 bg-gradient-to-r from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
          <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100">
            Open Positions
            <span className="ml-2 text-sm font-normal text-gray-500 dark:text-gray-400">
              ({positions.length})
            </span>
          </h2>
        </div>

        {loading ? (
          <div className="p-4 space-y-3">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="flex items-center gap-4 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg animate-pulse">
                <div className="h-5 w-20 bg-gray-200 dark:bg-gray-700 rounded" />
                <div className="h-5 w-12 bg-gray-200 dark:bg-gray-700 rounded ml-auto" />
                <div className="h-5 w-16 bg-gray-200 dark:bg-gray-700 rounded" />
                <div className="h-5 w-16 bg-gray-200 dark:bg-gray-700 rounded" />
                <div className="h-5 w-16 bg-gray-200 dark:bg-gray-700 rounded" />
                <div className="h-5 w-20 bg-gray-200 dark:bg-gray-700 rounded" />
                <div className="h-5 w-24 bg-gray-200 dark:bg-gray-700 rounded" />
                <div className="flex gap-2">
                  <div className="h-8 w-8 bg-gray-200 dark:bg-gray-700 rounded-lg" />
                  <div className="h-8 w-8 bg-gray-200 dark:bg-gray-700 rounded-lg" />
                  <div className="h-8 w-8 bg-gray-200 dark:bg-gray-700 rounded-lg" />
                </div>
              </div>
            ))}
          </div>
        ) : positions.length === 0 ? (
          <div className="p-16 text-center">
            <ArrowTrendingUpIcon className="h-12 w-12 mx-auto text-gray-200 dark:text-gray-700 mb-3" />
            <p className="text-gray-500 dark:text-gray-400 text-sm font-medium">
              No open positions
            </p>
            <p className="text-gray-400 dark:text-gray-500 text-xs mt-1">
              Run a trading cycle to open new positions
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50/50 dark:bg-gray-900/50">
                  <th className="px-4 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                    Symbol
                  </th>
                  <th className="px-4 py-3.5 text-right text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                    Qty
                  </th>
                  <th className="px-4 py-3.5 text-right text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                    Entry
                  </th>
                  <th className="px-4 py-3.5 text-right text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                    Stop Loss
                  </th>
                  <th className="px-4 py-3.5 text-right text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                    Target
                  </th>
                  <th className="px-4 py-3.5 text-right text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                    Investment
                  </th>
                  <th className="px-4 py-3.5 text-left text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                    Strategy
                  </th>
                  <th className="px-4 py-3.5 text-center text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700/50">
                {positions.map((pos) => {
                  const sl = pos.current_stop_loss || pos.stop_loss;
                  const target = pos.current_target || pos.target;
                  const currentPrice = pos.current_price || pos.entry_price;
                  const pnl = (currentPrice - pos.entry_price) * pos.quantity;
                  const pnlPct = pos.entry_price > 0 ? ((currentPrice - pos.entry_price) / pos.entry_price) * 100 : 0;

                  return (
                    <tr
                      key={pos.symbol}
                      className="hover:bg-gray-50/50 dark:hover:bg-gray-700/30 transition-colors"
                    >
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-gray-900 dark:text-gray-100">
                            {pos.symbol}
                          </span>
                          {pos.adds_count && pos.adds_count > 0 && (
                            <span className="px-1.5 py-0.5 text-[10px] font-semibold bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded">
                              +{pos.adds_count}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-4 text-right font-medium text-gray-900 dark:text-gray-100">
                        {pos.quantity}
                      </td>
                      <td className="px-4 py-4 text-right font-medium text-gray-900 dark:text-gray-100">
                        ₹{pos.entry_price?.toFixed(2)}
                      </td>
                      <td className="px-4 py-4 text-right font-medium text-red-600 dark:text-red-400">
                        ₹{sl?.toFixed(2)}
                      </td>
                      <td className="px-4 py-4 text-right font-medium text-green-600 dark:text-green-400">
                        ₹{target?.toFixed(2)}
                      </td>
                      <td className="px-4 py-4 text-right">
                        <div className="font-medium text-gray-900 dark:text-gray-100">
                          ₹{pos.total_investment?.toLocaleString('en-IN')}
                        </div>
                        {pnl !== 0 && (
                          <div
                            className={`text-xs font-medium ${
                              pnl >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
                            }`}
                          >
                            {pnl >= 0 ? '+' : ''}₹{pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 })} ({pnlPct.toFixed(1)}%)
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-4 text-gray-500 dark:text-gray-400 text-xs">
                        {pos.strategy_name}
                      </td>
                      <td className="px-4 py-4">
                        <div className="flex items-center justify-center gap-1.5">
                          <button
                            onClick={() => viewTimeline(pos.symbol)}
                            className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50 dark:hover:text-blue-400 dark:hover:bg-blue-900/30 rounded-lg transition"
                            title="View Timeline"
                          >
                            <EyeIcon className="h-4 w-4" />
                          </button>
                          <button
                            onClick={() => openEditModal(pos)}
                            className="p-2 text-gray-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:text-amber-400 dark:hover:bg-amber-900/30 rounded-lg transition"
                            title="Edit Entry Price"
                          >
                            <PencilIcon className="h-4 w-4" />
                          </button>
                          {pos.adds_count && pos.adds_count > 0 && (
                            <button
                              onClick={() => openPyramidModal(pos)}
                              className="p-2 text-gray-500 hover:text-violet-600 hover:bg-violet-50 dark:hover:text-violet-400 dark:hover:bg-violet-900/30 rounded-lg transition"
                              title={`Manage ${pos.adds_count} Pyramid(s)`}
                            >
                              <ArrowTrendingUpIcon className="h-4 w-4" />
                            </button>
                          )}
                          <button
                            onClick={() => handleClose(pos.symbol)}
                            className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-50 dark:hover:text-red-400 dark:hover:bg-red-900/30 rounded-lg transition"
                            title="Close Position"
                          >
                            <TrashIcon className="h-4 w-4" />
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

      {/* Edit Entry Price Modal */}
      {editModalPos && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 w-full max-w-md">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 dark:border-gray-700">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Update Entry Price
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                  {editModalPos.symbol}
                </p>
              </div>
              <button
                onClick={() => {
                  setEditModalPos(null);
                  setEntryPrice('');
                }}
                className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>

            <div className="p-6 space-y-5">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Entry Price (₹)
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={entryPrice}
                  onChange={(e) => setEntryPrice(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSaveEntryPrice()}
                  className="w-full px-4 py-3 border border-gray-200 dark:border-gray-600 rounded-xl bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 text-lg font-semibold focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition"
                  placeholder="0.00"
                  autoFocus
                />
                <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
                  Current: ₹{editModalPos.entry_price?.toFixed(2)}
                  {editModalPos.adds_count && editModalPos.adds_count > 0 && (
                    <span className="ml-2 text-blue-600 dark:text-blue-400">
                      ({editModalPos.adds_count} pyramid adds)
                    </span>
                  )}
                </p>
              </div>

              {editModalPos.adds_count && editModalPos.adds_count > 0 && (
                <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                  <p className="text-xs text-blue-700 dark:text-blue-300">
                    <span className="font-semibold">Note:</span> This position has{' '}
                    {editModalPos.adds_count} pyramid addition(s). Updating the entry price will
                    recalculate the average entry for all shares.
                  </p>
                </div>
              )}

              <div className="flex gap-3 pt-2">
                <button
                  onClick={handleSaveEntryPrice}
                  className="flex-1 inline-flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 font-medium transition"
                >
                  <CheckIcon className="h-4 w-4" />
                  Save
                </button>
                <button
                  onClick={() => {
                    setEditModalPos(null);
                    setEntryPrice('');
                  }}
                  className="px-6 py-3 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-xl hover:bg-gray-200 dark:hover:bg-gray-600 font-medium transition"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Pyramid Management Modal */}
      {pyramidModalPos && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 w-full max-w-lg">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 dark:border-gray-700">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Manage Pyramid Prices
                </h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
                  {pyramidModalPos.symbol} · {pyramidModalPos.quantity} total shares
                </p>
              </div>
              <button
                onClick={() => {
                  setPyramidModalPos(null);
                  setPyramidEdits([]);
                }}
                className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div className="p-3 bg-violet-50 dark:bg-violet-900/20 rounded-lg border border-violet-200 dark:border-violet-800">
                <p className="text-xs text-violet-700 dark:text-violet-300">
                  <span className="font-semibold">How it works:</span> Edit each pyramid's actual execution price from your broker. The system will recalculate the weighted average entry price.
                </p>
              </div>

              <div className="space-y-3">
                {pyramidEdits.map((edit, idx) => {
                  const pyramidUpdate = (pyramidModalPos.updates || []).filter((u: any) => u.type === 'PYRAMID')[idx];
                  const date = pyramidUpdate?.date ? new Date(pyramidUpdate.date).toLocaleDateString('en-IN') : 'Unknown';
                  
                  return (
                    <div key={idx} className="p-4 bg-gray-50 dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-700">
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                            Pyramid #{idx + 1}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">{date}</p>
                        </div>
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                          {edit.qty} shares
                        </span>
                      </div>
                      <div>
                        <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1.5">
                          Execution Price (₹)
                        </label>
                        <input
                          type="number"
                          step="0.01"
                          value={edit.price}
                          onChange={(e) => {
                            const newEdits = [...pyramidEdits];
                            newEdits[idx] = { ...newEdits[idx], price: e.target.value };
                            setPyramidEdits(newEdits);
                          }}
                          className="w-full px-3 py-2 border border-gray-200 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 font-semibold focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent transition"
                          placeholder="0.00"
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="pt-2 flex gap-3">
                <button
                  onClick={handleSavePyramids}
                  className="flex-1 inline-flex items-center justify-center gap-2 px-6 py-3 bg-violet-600 text-white rounded-xl hover:bg-violet-700 font-medium transition"
                >
                  <CheckIcon className="h-4 w-4" />
                  Recalculate & Save
                </button>
                <button
                  onClick={() => {
                    setPyramidModalPos(null);
                    setPyramidEdits([]);
                  }}
                  className="px-6 py-3 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-xl hover:bg-gray-200 dark:hover:bg-gray-600 font-medium transition"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function StatCard({
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
              positive ? 'text-green-500' : 'text-red-500'
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
