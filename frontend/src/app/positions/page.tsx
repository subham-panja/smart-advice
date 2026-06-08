'use client';

import { useState, useEffect } from 'react';
import {
  PencilIcon,
  XMarkIcon,
  TrashIcon,
  CheckIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import {
  getPositions,
  updatePosition,
  closePosition,
  Position,
} from '@/lib/api';

interface EditForm {
  quantity: string;
  entry_price: string;
  current_stop_loss: string;
  current_target: string;
  total_investment: string;
  adds_count: string;
  targets_hit: string;
}

const emptyEditForm: EditForm = {
  quantity: '',
  entry_price: '',
  current_stop_loss: '',
  current_target: '',
  total_investment: '',
  adds_count: '',
  targets_hit: '',
};

export default function PositionsPage() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [editModalPos, setEditModalPos] = useState<Position | null>(null);
  const [editForm, setEditForm] = useState<EditForm>(emptyEditForm);
  const [actionMsg, setActionMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const fetchData = async () => {
    setLoading(true);
    const res = await getPositions('OPEN');
    if (res.status === 'success') setPositions(res.positions);
    setLoading(false);
  };

  useEffect(() => { fetchData(); }, []);

  const openEditModal = (pos: Position) => {
    setEditModalPos(pos);
    setEditForm({
      quantity: String(pos.quantity || ''),
      entry_price: String(pos.entry_price || ''),
      current_stop_loss: String(pos.current_stop_loss || pos.stop_loss || ''),
      current_target: String(pos.current_target || pos.target || ''),
      total_investment: String(pos.total_investment || ''),
      adds_count: String(pos.adds_count || 0),
      targets_hit: String(pos.targets_hit || pos.current_target_idx || 0),
    });
  };

  const handleSaveEdit = async () => {
    if (!editModalPos) return;
    const data: Record<string, any> = {};
    if (editForm.quantity) data.quantity = parseInt(editForm.quantity);
    if (editForm.entry_price) data.entry_price = parseFloat(editForm.entry_price);
    if (editForm.current_stop_loss) data.current_stop_loss = parseFloat(editForm.current_stop_loss);
    if (editForm.current_target) data.current_target = parseFloat(editForm.current_target);
    if (editForm.total_investment) data.total_investment = parseFloat(editForm.total_investment);
    if (editForm.adds_count) data.adds_count = parseInt(editForm.adds_count);
    if (editForm.targets_hit) data.targets_hit = parseInt(editForm.targets_hit);

    const res = await updatePosition(editModalPos.symbol, data);
    if (res.status === 'success') {
      setActionMsg({ type: 'success', text: `${editModalPos.symbol} updated!` });
      setEditModalPos(null);
      setEditForm(emptyEditForm);
      fetchData();
    } else {
      setActionMsg({ type: 'error', text: res.error || 'Update failed' });
    }
  };

  const handleClose = async (symbol: string) => {
    if (!confirm(`Close position for ${symbol}?`)) return;
    const res = await closePosition(symbol);
    if (res.status === 'success') {
      setActionMsg({ type: 'success', text: `${symbol} closed!` });
      fetchData();
    } else {
      setActionMsg({ type: 'error', text: res.error || 'Close failed' });
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Positions</h1>
        <button onClick={fetchData} className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700">
          <ArrowPathIcon className="h-5 w-5" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Action Message */}
      {actionMsg && (
        <div className={`p-4 rounded-lg ${actionMsg.type === 'success' ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200' : 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200'}`}>
          {actionMsg.text}
        </div>
      )}

      {/* Open Positions Table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Open Positions ({positions.length})</h2>
        </div>
        {loading ? (
          <div className="p-8 text-center text-gray-500">Loading...</div>
        ) : positions.length === 0 ? (
          <div className="p-8 text-center text-gray-500">No open positions.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Symbol</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Qty</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Entry</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">SL</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Target</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Investment</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Strategy</th>
                  <th className="px-4 py-3 text-left font-medium text-gray-700 dark:text-gray-300">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {positions.map((pos) => (
                  <tr key={pos.symbol} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="px-4 py-3 font-semibold text-gray-900 dark:text-gray-100">{pos.symbol}</td>
                    <td className="px-4 py-3 text-gray-700 dark:text-gray-300">{pos.quantity}</td>
                    <td className="px-4 py-3 text-gray-700 dark:text-gray-300">₹{pos.entry_price?.toFixed(2)}</td>
                    <td className="px-4 py-3 text-red-600 dark:text-red-400">₹{(pos.current_stop_loss || pos.stop_loss)?.toFixed(2)}</td>
                    <td className="px-4 py-3 text-green-600 dark:text-green-400">₹{(pos.current_target || pos.target)?.toFixed(2)}</td>
                    <td className="px-4 py-3 text-gray-700 dark:text-gray-300">₹{pos.total_investment?.toLocaleString()}</td>
                    <td className="px-4 py-3 text-gray-600 dark:text-gray-400">{pos.strategy_name}</td>
                    <td className="px-4 py-3 flex gap-2">
                      <button onClick={() => openEditModal(pos)} className="p-1.5 text-blue-600 hover:bg-blue-100 dark:hover:bg-blue-900/30 rounded" title="Edit">
                        <PencilIcon className="h-5 w-5" />
                      </button>
                      <button onClick={() => handleClose(pos.symbol)} className="p-1.5 text-red-600 hover:bg-red-100 dark:hover:bg-red-900/30 rounded" title="Close">
                        <TrashIcon className="h-5 w-5" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Edit Modal */}
      {editModalPos && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-2xl border border-gray-200 dark:border-gray-700 w-full max-w-lg mx-4">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Edit Position</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400">{editModalPos.symbol}</p>
              </div>
              <button
                onClick={() => { setEditModalPos(null); setEditForm(emptyEditForm); }}
                className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Quantity</label>
                  <input type="number" value={editForm.quantity} onChange={(e) => setEditForm({ ...editForm, quantity: e.target.value })} className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Entry Price</label>
                  <input type="number" step="0.01" value={editForm.entry_price} onChange={(e) => setEditForm({ ...editForm, entry_price: e.target.value })} className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Stop Loss</label>
                  <input type="number" step="0.01" value={editForm.current_stop_loss} onChange={(e) => setEditForm({ ...editForm, current_stop_loss: e.target.value })} className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Target</label>
                  <input type="number" step="0.01" value={editForm.current_target} onChange={(e) => setEditForm({ ...editForm, current_target: e.target.value })} className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Total Investment</label>
                  <input type="number" step="0.01" value={editForm.total_investment} onChange={(e) => setEditForm({ ...editForm, total_investment: e.target.value })} className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Adds Count</label>
                  <input type="number" value={editForm.adds_count} onChange={(e) => setEditForm({ ...editForm, adds_count: e.target.value })} className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Targets Hit</label>
                  <input type="number" value={editForm.targets_hit} onChange={(e) => setEditForm({ ...editForm, targets_hit: e.target.value })} className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100" />
                </div>
              </div>

              <div className="flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
                <button onClick={handleSaveEdit} className="flex items-center space-x-2 px-6 py-2.5 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition">
                  <CheckIcon className="h-4 w-4" />
                  <span>Save Changes</span>
                </button>
                <button onClick={() => { setEditModalPos(null); setEditForm(emptyEditForm); }} className="px-6 py-2.5 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition">
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
