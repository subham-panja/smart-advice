'use client';

import {
  ChartBarIcon,
} from '@heroicons/react/24/outline';
import { Bar, Doughnut } from 'react-chartjs-2';
import { useEffect, useState } from 'react';
import { getRecommendations, StockRecommendation, getCycleStats, CycleStats } from '@/lib/api';
import '../lib/chartConfig';

export default function Home() {
  const [recommendations, setRecommendations] = useState<StockRecommendation[]>([]);
  const [topN, setTopN] = useState(10);
  const [isClient, setIsClient] = useState(false);
  const [stats, setStats] = useState<Partial<CycleStats>>({});

  useEffect(() => {
    setIsClient(true);
    getCycleStats().then(res => {
      if (res.status === 'success') setStats(res);
    });
  }, []);

  useEffect(() => {
    async function fetchData() {
      const response = await getRecommendations();
      if (response.status === 'success' && response.recommendations) {
        setRecommendations(response.recommendations);
      }
    }
    fetchData();
  }, []);

  if (!isClient) {
    return <div className="flex items-center justify-center min-h-screen">
      <div className="text-xl text-gray-600 dark:text-gray-300">Loading...</div>
    </div>;
  }

  const topStocks = recommendations.slice(0, topN);

  const backtestReturnsData = {
    labels: topStocks.map(rec => rec.symbol),
    datasets: [{
      label: 'Backtest CAGR (%)',
      data: topStocks.map(rec => rec.backtest_cagr || 0),
      backgroundColor: 'rgba(34, 197, 94, 0.6)',
      borderColor: 'rgba(34, 197, 94, 1)',
      borderWidth: 2,
    }],
  };

  const profitPercentageData = {
    labels: topStocks.map(rec => rec.symbol),
    datasets: [{
      label: 'Combined Score',
      data: topStocks.map(rec => rec.combined_score),
      backgroundColor: 'rgba(59, 130, 246, 0.6)',
      borderColor: 'rgba(59, 130, 246, 1)',
      borderWidth: 2,
    }],
  };

  const scoresComparisonData = {
    labels: topStocks.map(rec => rec.symbol),
    datasets: [
      { label: 'Technical Score', data: topStocks.map(rec => rec.technical_score), backgroundColor: 'rgba(251, 191, 36, 0.6)', borderColor: 'rgba(251, 191, 36, 1)', borderWidth: 2 },
      { label: 'Fundamental Score', data: topStocks.map(rec => rec.fundamental_score), backgroundColor: 'rgba(139, 92, 246, 0.6)', borderColor: 'rgba(139, 92, 246, 1)', borderWidth: 2 },
    ],
  };

  const strengthCounts = recommendations.reduce((acc, rec) => {
    acc[rec.recommendation_strength] = (acc[rec.recommendation_strength] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const doughnutData = {
    labels: Object.keys(strengthCounts),
    datasets: [{
      data: Object.values(strengthCounts),
      backgroundColor: ['rgba(34, 197, 94, 0.8)', 'rgba(59, 130, 246, 0.8)', 'rgba(251, 191, 36, 0.8)', 'rgba(239, 68, 68, 0.8)', 'rgba(139, 92, 246, 0.8)'],
      borderColor: ['rgba(34, 197, 94, 1)', 'rgba(59, 130, 246, 1)', 'rgba(251, 191, 36, 1)', 'rgba(239, 68, 68, 1)', 'rgba(139, 92, 246, 1)'],
      borderWidth: 2,
    }],
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8 md:space-y-12">
      {/* Hero */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center p-3 bg-blue-100 dark:bg-blue-900/30 rounded-full mb-6">
          <ChartBarIcon className="h-12 w-12 text-blue-600 dark:text-blue-400" />
        </div>
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Stock Advice Dashboard
        </h1>
        <p className="text-lg sm:text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto px-4">
          AI-powered stock analysis and recommendations for smart investing.
        </p>
      </div>

      {/* Portfolio Summary */}
      {stats.open_positions !== undefined && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-sm text-gray-500 dark:text-gray-400">Open Positions</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">{stats.open_positions}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-sm text-gray-500 dark:text-gray-400">Total Equity</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">₹{stats.total_equity?.toLocaleString()}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-sm text-gray-500 dark:text-gray-400">Cash Available</p>
            <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">₹{stats.cash_remaining?.toLocaleString()}</p>
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700 p-4">
            <p className="text-sm text-gray-500 dark:text-gray-400">Total PnL</p>
            <p className={`text-2xl font-bold ${(stats.pnl_pct || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>{stats.pnl_pct?.toFixed(2)}%</p>
          </div>
        </div>
      )}

      {/* Charts */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6 lg:p-8 border border-gray-200 dark:border-gray-700">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6">
          <h3 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">Top Stocks Analysis</h3>
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Show Top:</label>
            <select value={topN} onChange={(e) => setTopN(Number(e.target.value))} className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100">
              {Array.from({ length: 20 }, (_, i) => i + 1).map(num => (<option key={num} value={num}>{num}</option>))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 text-center">Best Backtest Returns (CAGR %)</h4>
            <div className="h-64"><Bar data={backtestReturnsData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }} /></div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 text-center">Combined Scores</h4>
            <div className="h-64"><Bar data={profitPercentageData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }} /></div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 text-center">Technical vs Fundamental</h4>
            <div className="h-64"><Bar data={scoresComparisonData} options={{ responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } } }} /></div>
          </div>
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 text-center">Recommendation Distribution</h4>
            <div className="h-64 flex items-center justify-center"><div className="w-48 h-48"><Doughnut data={doughnutData} options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' as const, labels: { boxWidth: 12, padding: 8, font: { size: 10 } } } } }} /></div></div>
          </div>
        </div>
      </div>
    </div>
  );
}
