'use client';

import { useState, useEffect } from 'react';
import {
  PlayIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  CalendarIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline';
import { runOrchestrator, OrchestratorConfig, getOrchestratorStatus } from '@/lib/api';
import Terminal from '../components/Terminal';

const API_HOST = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5001';

export default function TradingCyclePage() {
  const [mode, setMode] = useState<'live' | 'replay' | 'date'>('live');
  const [replayDays, setReplayDays] = useState<string>('');
  const [date, setDate] = useState('');
  const [verbose, setVerbose] = useState(false);
  const [showTerminal, setShowTerminal] = useState(false);
  const [status, setStatus] = useState<{ type: 'idle' | 'loading' | 'success' | 'error'; message: string }>({
    type: 'idle',
    message: '',
  });

  useEffect(() => {
    const poll = setInterval(async () => {
      const s = await getOrchestratorStatus();
      if (s.status === 'completed' || s.status === 'error') {
        setStatus((prev) => {
          if (prev.type === 'loading') {
            setShowTerminal(false);
            return { type: s.status as 'success' | 'error', message: s.message };
          }
          return prev;
        });
      }
    }, 3000);
    return () => clearInterval(poll);
  }, []);

  const handleRun = async () => {
    setStatus({ type: 'loading', message: 'Starting trading cycle...' });
    setShowTerminal(true);

    const config: OrchestratorConfig = { mode, verbose };
    if (mode === 'replay' && replayDays) config.replay_days = parseInt(replayDays);
    if (mode === 'date') config.date = date;

    const response = await runOrchestrator(config);
    if (response.status === 'error') {
      setStatus({ type: 'error', message: response.error || 'Failed to start' });
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto space-y-8">
      {/* Hero */}
      <div className="bg-gradient-to-r from-green-600 to-blue-700 rounded-2xl p-8 text-white">
        <div className="flex items-center space-x-4 mb-4">
          <div className="p-3 bg-white/20 rounded-xl">
            <ArrowTrendingUpIcon className="h-8 w-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Run Trading Cycle</h1>
            <p className="text-green-100 mt-1">
              Monitor positions, scan for new buys, and execute trades
            </p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <ArrowTrendingUpIcon className="h-5 w-5 text-green-200" />
              <span className="text-sm font-medium text-green-100">Phase 1: Monitor</span>
            </div>
            <p className="text-xs text-green-200 mt-1">Trail SLs, hit targets, time stops</p>
          </div>
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <ArrowTrendingUpIcon className="h-5 w-5 text-green-200" />
              <span className="text-sm font-medium text-green-100">Phase 2: Analyze</span>
            </div>
            <p className="text-xs text-green-200 mt-1">Scan 200+ stocks via screener</p>
          </div>
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <ArrowTrendingUpIcon className="h-5 w-5 text-green-200" />
              <span className="text-sm font-medium text-green-100">Phase 3: Execute</span>
            </div>
            <p className="text-xs text-green-200 mt-1">Buy recommended stocks with position sizing</p>
          </div>
        </div>
      </div>

      {/* Config */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 px-8 py-6">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Cycle Configuration</h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">Choose how to run the trading cycle</p>
        </div>

        <div className="p-8 space-y-8">
          {/* Mode Selection */}
          <div className="grid md:grid-cols-3 gap-4">
            <label className={`group flex flex-col items-center p-6 border-2 rounded-xl cursor-pointer transition-all ${
              mode === 'live' ? 'border-green-500 bg-green-50 dark:bg-green-900/20' : 'border-gray-200 dark:border-gray-600 hover:border-green-300'
            }`}>
              <ArrowTrendingUpIcon className="h-8 w-8 mb-2 text-green-600 dark:text-green-400" />
              <span className="font-semibold text-gray-900 dark:text-gray-100">Paper / Live</span>
              <span className="text-xs text-gray-500 dark:text-gray-400 mt-1 text-center">Run with current config</span>
              <input type="radio" name="mode" checked={mode === 'live'} onChange={() => setMode('live')} className="mt-3" />
            </label>

            <label className={`group flex flex-col items-center p-6 border-2 rounded-xl cursor-pointer transition-all ${
              mode === 'replay' ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-600 hover:border-blue-300'
            }`}>
              <ClockIcon className="h-8 w-8 mb-2 text-blue-600 dark:text-blue-400" />
              <span className="font-semibold text-gray-900 dark:text-gray-100">Replay</span>
              <span className="text-xs text-gray-500 dark:text-gray-400 mt-1 text-center">Replay last N trading days</span>
              <input type="radio" name="mode" checked={mode === 'replay'} onChange={() => setMode('replay')} className="mt-3" />
            </label>

            <label className={`group flex flex-col items-center p-6 border-2 rounded-xl cursor-pointer transition-all ${
              mode === 'date' ? 'border-purple-500 bg-purple-50 dark:bg-purple-900/20' : 'border-gray-200 dark:border-gray-600 hover:border-purple-300'
            }`}>
              <CalendarIcon className="h-8 w-8 mb-2 text-purple-600 dark:text-purple-400" />
              <span className="font-semibold text-gray-900 dark:text-gray-100">Specific Date</span>
              <span className="text-xs text-gray-500 dark:text-gray-400 mt-1 text-center">Run for a single date</span>
              <input type="radio" name="mode" checked={mode === 'date'} onChange={() => setMode('date')} className="mt-3" />
            </label>
          </div>

          {/* Mode-specific fields */}
          {mode === 'replay' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Replay Days</label>
              <input
                type="number"
                min="1"
                max="60"
                placeholder="e.g. 5"
                value={replayDays}
                onChange={(e) => setReplayDays(e.target.value)}
                className="w-full md:w-48 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Number of trading days to replay (default: 5)</p>
            </div>
          )}

          {mode === 'date' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Date</label>
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="w-full md:w-48 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
            </div>
          )}

          {/* Options */}
          <div className="space-y-3">
            <label className="flex items-center space-x-3 cursor-pointer">
              <input
                type="checkbox"
                checked={verbose}
                onChange={(e) => setVerbose(e.target.checked)}
                className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
              />
              <span className="text-gray-900 dark:text-gray-100 font-medium">Verbose Logging</span>
              <span className="text-sm text-gray-500 dark:text-gray-400">(show all logs in terminal)</span>
            </label>
          </div>

          {/* Status */}
          {status.type !== 'idle' && (
            <div className={`p-6 rounded-xl border-2 ${
              status.type === 'loading' ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800' :
              status.type === 'success' ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' :
              'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
            }`}>
              <div className="flex items-center space-x-4">
                {status.type === 'loading' && <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600" />}
                {status.type === 'success' && <CheckCircleIcon className="h-6 w-6 text-green-600" />}
                {status.type === 'error' && <ExclamationTriangleIcon className="h-6 w-6 text-red-600" />}
                <p className={`font-semibold ${
                  status.type === 'loading' ? 'text-blue-800 dark:text-blue-200' :
                  status.type === 'success' ? 'text-green-800 dark:text-green-200' :
                  'text-red-800 dark:text-red-200'
                }`}>{status.message}</p>
              </div>
            </div>
          )}

          {/* Run Button */}
          <div className="pt-6 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={handleRun}
              disabled={status.type === 'loading'}
              className="flex items-center space-x-2 px-8 py-3 text-white bg-gradient-to-r from-green-600 to-blue-600 rounded-xl hover:from-green-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transition-all"
            >
              {status.type === 'loading' ? (
                <><div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" /><span>Running...</span></>
              ) : (
                <><PlayIcon className="h-5 w-5" /><span>Run Trading Cycle</span></>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Terminal Modal */}
      <Terminal isOpen={showTerminal} onClose={() => setShowTerminal(false)} apiHost={API_HOST} />
    </div>
  );
}
