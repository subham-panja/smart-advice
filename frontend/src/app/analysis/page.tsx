'use client';

import { useState } from 'react';
import { 
  PlayIcon, 
  ExclamationTriangleIcon, 
  CheckCircleIcon, 
  ChartBarIcon,
  CogIcon,
  SparklesIcon,
  ClockIcon,
  ServerIcon
} from '@heroicons/react/24/outline';
import { triggerAnalysis, AnalysisConfig } from '@/lib/api';

interface AnalysisStatus {
  type: 'idle' | 'loading' | 'success' | 'error';
  message: string;
}

export default function AnalysisPage() {
  const [config, setConfig] = useState<AnalysisConfig>({
    max_stocks: undefined,
    test: false,
    all: false,
    offline: false,
    verbose: false,
    purge_days: undefined,
    disable_volume_filter: false,
  });

  const [status, setStatus] = useState<AnalysisStatus>({
    type: 'idle',
    message: '',
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    setStatus({ type: 'loading', message: 'Starting analysis...' });
    
    try {
      const response = await triggerAnalysis(config);
      
      if (response.status === 'success') {
        setStatus({
          type: 'success',
          message: response.message || 'Analysis started successfully!',
        });
      } else {
        setStatus({
          type: 'error',
          message: response.error || 'Failed to start analysis',
        });
      }
    } catch {
      setStatus({
        type: 'error',
        message: 'Failed to connect to the backend. Please check if the server is running.',
      });
    }
  };

  const handleReset = () => {
    setConfig({
      max_stocks: undefined,
      test: false,
      all: false,
      offline: false,
      verbose: false,
      purge_days: undefined,
      disable_volume_filter: false,
    });
    setStatus({ type: 'idle', message: '' });
  };

  return (
    <div className="w-full max-w-7xl mx-auto space-y-8">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-700 rounded-2xl p-8 text-white">
        <div className="flex items-center space-x-4 mb-4">
          <div className="p-3 bg-white/20 rounded-xl">
            <ChartBarIcon className="h-8 w-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold">Generate Stock Analysis</h1>
            <p className="text-blue-100 mt-1">
              Configure analysis parameters and trigger comprehensive stock analysis
            </p>
          </div>
        </div>
        
        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <SparklesIcon className="h-5 w-5 text-blue-200" />
              <span className="text-sm font-medium text-blue-100">AI-Powered</span>
            </div>
            <p className="text-xs text-blue-200 mt-1">Advanced machine learning algorithms</p>
          </div>
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <ClockIcon className="h-5 w-5 text-blue-200" />
              <span className="text-sm font-medium text-blue-100">Real-time</span>
            </div>
            <p className="text-xs text-blue-200 mt-1">Live market data analysis</p>
          </div>
          <div className="bg-white/10 rounded-lg p-4">
            <div className="flex items-center space-x-2">
              <ServerIcon className="h-5 w-5 text-blue-200" />
              <span className="text-sm font-medium text-blue-100">Scalable</span>
            </div>
            <p className="text-xs text-blue-200 mt-1">Process thousands of stocks</p>
          </div>
        </div>
      </div>

      {/* Main Form */}
      <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl border border-gray-200 dark:border-gray-700 overflow-hidden">
        <div className="border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 px-8 py-6">
          <div className="flex items-center space-x-3">
            <CogIcon className="h-6 w-6 text-gray-600 dark:text-gray-400" />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              Analysis Configuration
            </h2>
          </div>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Customize your analysis parameters for optimal results
          </p>
        </div>

        <form onSubmit={handleSubmit} className="p-8 space-y-8">
          {/* Numeric Inputs */}
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center space-x-2">
              <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
              <span>Parameters</span>
            </h3>
            
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label htmlFor="maxStocks" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Max Stocks
                </label>
                <div className="relative">
                  <input
                    type="number"
                    id="maxStocks"
                    min="1"
                    placeholder="Leave empty for all stocks"
                    value={config.max_stocks || ''}
                    onChange={(e) => setConfig({
                      ...config,
                      max_stocks: e.target.value ? parseInt(e.target.value) : undefined
                    })}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                  />
                </div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Limit the number of stocks to analyze (useful for testing)
                </p>
              </div>

              <div className="space-y-2">
                <label htmlFor="purgeDays" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Purge Days
                </label>
                <div className="relative">
                  <input
                    type="number"
                    id="purgeDays"
                    min="0"
                    placeholder="7 (default)"
                    value={config.purge_days || ''}
                    onChange={(e) => setConfig({
                      ...config,
                      purge_days: e.target.value ? parseInt(e.target.value) : undefined
                    })}
                    className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-xl bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-colors"
                  />
                </div>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Days to keep old data (0 = remove all data)
                </p>
              </div>
            </div>
          </div>

          {/* Analysis Options */}
          <div className="space-y-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center space-x-2">
              <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
              <span>Analysis Options</span>
            </h3>
            
            <div className="grid md:grid-cols-2 gap-4">
              {/* Test Mode */}
              <label className="group relative flex items-center justify-between p-4 border-2 border-gray-200 dark:border-gray-600 rounded-xl hover:border-blue-300 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 cursor-pointer transition-all duration-200">
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg group-hover:bg-yellow-200 dark:group-hover:bg-yellow-900/50 transition-colors">
                    <SparklesIcon className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
                  </div>
                  <div>
                    <span className="font-medium text-gray-900 dark:text-gray-100">Test Mode</span>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Run with limited stocks for testing</p>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={config.test}
                  onChange={(e) => setConfig({ ...config, test: e.target.checked })}
                  className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded transition-colors"
                />
              </label>

              {/* All Symbols */}
              <label className="group relative flex items-center justify-between p-4 border-2 border-gray-200 dark:border-gray-600 rounded-xl hover:border-blue-300 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 cursor-pointer transition-all duration-200">
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg group-hover:bg-green-200 dark:group-hover:bg-green-900/50 transition-colors">
                    <ChartBarIcon className="h-4 w-4 text-green-600 dark:text-green-400" />
                  </div>
                  <div>
                    <span className="font-medium text-gray-900 dark:text-gray-100">All Symbols</span>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Include all NSE symbols (including inactive)</p>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={config.all}
                  onChange={(e) => setConfig({ ...config, all: e.target.checked })}
                  className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded transition-colors"
                />
              </label>

              {/* Offline Mode */}
              <label className="group relative flex items-center justify-between p-4 border-2 border-gray-200 dark:border-gray-600 rounded-xl hover:border-blue-300 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 cursor-pointer transition-all duration-200">
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg group-hover:bg-gray-200 dark:group-hover:bg-gray-600 transition-colors">
                    <ServerIcon className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  </div>
                  <div>
                    <span className="font-medium text-gray-900 dark:text-gray-100">Offline Mode</span>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Use cached data only, no API calls</p>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={config.offline}
                  onChange={(e) => setConfig({ ...config, offline: e.target.checked })}
                  className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded transition-colors"
                />
              </label>

              {/* Verbose Mode */}
              <label className="group relative flex items-center justify-between p-4 border-2 border-gray-200 dark:border-gray-600 rounded-xl hover:border-blue-300 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 cursor-pointer transition-all duration-200">
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg group-hover:bg-blue-200 dark:group-hover:bg-blue-900/50 transition-colors">
                    <CogIcon className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <span className="font-medium text-gray-900 dark:text-gray-100">Verbose Mode</span>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Enable detailed logging output</p>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={config.verbose}
                  onChange={(e) => setConfig({ ...config, verbose: e.target.checked })}
                  className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded transition-colors"
                />
              </label>

              {/* Disable Volume Filter */}
              <label className="group relative flex items-center justify-between p-4 border-2 border-gray-200 dark:border-gray-600 rounded-xl hover:border-blue-300 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 cursor-pointer transition-all duration-200 md:col-span-2">
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-red-100 dark:bg-red-900/30 rounded-lg group-hover:bg-red-200 dark:group-hover:bg-red-900/50 transition-colors">
                    <ExclamationTriangleIcon className="h-4 w-4 text-red-600 dark:text-red-400" />
                  </div>
                  <div>
                    <span className="font-medium text-gray-900 dark:text-gray-100">Disable Volume Filter</span>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Skip volume-based filtering (may include low-volume stocks)</p>
                  </div>
                </div>
                <input
                  type="checkbox"
                  checked={config.disable_volume_filter}
                  onChange={(e) => setConfig({ ...config, disable_volume_filter: e.target.checked })}
                  className="h-5 w-5 text-blue-600 focus:ring-blue-500 border-gray-300 rounded transition-colors"
                />
              </label>
            </div>
          </div>

          {/* Status Display */}
          {status.type !== 'idle' && (
            <div className={`p-6 rounded-xl border-2 ${
              status.type === 'loading' 
                ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800' 
                : status.type === 'success' 
                ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800' 
                : 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
            }`}>
              <div className="flex items-center space-x-4">
                {status.type === 'loading' && (
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 dark:border-blue-400"></div>
                )}
                {status.type === 'success' && (
                  <CheckCircleIcon className="h-6 w-6 text-green-600 dark:text-green-400" />
                )}
                {status.type === 'error' && (
                  <ExclamationTriangleIcon className="h-6 w-6 text-red-600 dark:text-red-400" />
                )}
                <div>
                  <p className={`font-semibold ${
                    status.type === 'loading' 
                      ? 'text-blue-800 dark:text-blue-200' 
                      : status.type === 'success' 
                      ? 'text-green-800 dark:text-green-200' 
                      : 'text-red-800 dark:text-red-200'
                  }`}>
                    {status.type === 'loading' ? 'Analysis in Progress' : 
                     status.type === 'success' ? 'Analysis Started' : 'Analysis Failed'}
                  </p>
                  <p className={`text-sm ${
                    status.type === 'loading' 
                      ? 'text-blue-700 dark:text-blue-300' 
                      : status.type === 'success' 
                      ? 'text-green-700 dark:text-green-300' 
                      : 'text-red-700 dark:text-red-300'
                  }`}>
                    {status.message}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 pt-6 border-t border-gray-200 dark:border-gray-700">
            <button
              type="button"
              onClick={handleReset}
              className="flex-1 sm:flex-none px-6 py-3 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-xl hover:bg-gray-200 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800 transition-colors"
            >
              Reset Configuration
            </button>
            
            <button
              type="submit"
              disabled={status.type === 'loading'}
              className="flex-1 sm:flex-none flex items-center justify-center space-x-2 px-8 py-3 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 border border-transparent rounded-xl hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl"
            >
              {status.type === 'loading' ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <PlayIcon className="h-5 w-5" />
                  <span>Start Analysis</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
