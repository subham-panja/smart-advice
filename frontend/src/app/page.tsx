'use client';

import {
  ChartBarIcon,
  PlayIcon,
  ArrowTrendingUpIcon,
  SparklesIcon,
  ShieldCheckIcon,
  ClockIcon,
  CogIcon,
  TrendingUpIcon,
  DocumentTextIcon,
  CpuChipIcon
} from '@heroicons/react/24/outline';
import { Bar, Line, Doughnut } from 'react-chartjs-2';
import { useEffect, useState } from 'react';
import { getRecommendations, StockRecommendation } from '@/lib/api';
import '../lib/chartConfig';
import ApiTest from './components/ApiTest';

export default function Home() {
  const [recommendations, setRecommendations] = useState<StockRecommendation[]>([]);
  const [topN, setTopN] = useState(10); // State for top N stocks selection
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  useEffect(() => {
    async function fetchData() {
      console.log('Fetching recommendations...');
      const response = await getRecommendations();
      console.log('API Response:', response);
      console.log('Response status:', response.status);
      console.log('Recommendations:', response.recommendations);
      console.log('Recommendations length:', response.recommendations?.length);
      if (response.status === 'success' && response.recommendations) {
        console.log('Setting recommendations:', response.recommendations);
        setRecommendations(response.recommendations);
      } else {
        console.log('NOT setting recommendations - status:', response.status);
      }
    }
    fetchData();
  }, []);

  if (!isClient) {
    return <div className="flex items-center justify-center min-h-screen">
      <div className="text-xl text-gray-600 dark:text-gray-300">Loading...</div>
    </div>;
  }

  // Chart data preparation
  const topStocks = recommendations.slice(0, topN); // Show top N stocks in charts

  // Chart 1: Top stocks by backtest returns
  const backtestReturnsData = {
    labels: topStocks.map(rec => rec.symbol),
    datasets: [
      {
        label: 'Backtest CAGR (%)',
        data: topStocks.map(rec => rec.backtest_cagr || 0),
        backgroundColor: 'rgba(34, 197, 94, 0.6)',
        borderColor: 'rgba(34, 197, 94, 1)',
        borderWidth: 2,
      },
    ],
  };

  // Chart 2: Profit percentage of top stocks
  const profitPercentageData = {
    labels: topStocks.map(rec => rec.symbol),
    datasets: [
      {
        label: 'Combined Score',
        data: topStocks.map(rec => rec.combined_score),
        backgroundColor: 'rgba(59, 130, 246, 0.6)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2,
      },
    ],
  };

  // Chart 3: Technical vs Fundamental scores
  const scoresComparisonData = {
    labels: topStocks.map(rec => rec.symbol),
    datasets: [
      {
        label: 'Technical Score',
        data: topStocks.map(rec => rec.technical_score),
        backgroundColor: 'rgba(251, 191, 36, 0.6)',
        borderColor: 'rgba(251, 191, 36, 1)',
        borderWidth: 2,
      },
      {
        label: 'Fundamental Score',
        data: topStocks.map(rec => rec.fundamental_score),
        backgroundColor: 'rgba(139, 92, 246, 0.6)',
        borderColor: 'rgba(139, 92, 246, 1)',
        borderWidth: 2,
      },
    ],
  };

  // Recommendation strength distribution
  const strengthCounts = recommendations.reduce((acc, rec) => {
    acc[rec.recommendation_strength] = (acc[rec.recommendation_strength] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const doughnutData = {
    labels: Object.keys(strengthCounts),
    datasets: [
      {
        data: Object.values(strengthCounts),
        backgroundColor: [
          'rgba(34, 197, 94, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(251, 191, 36, 0.8)',
          'rgba(239, 68, 68, 0.8)',
          'rgba(139, 92, 246, 0.8)',
        ],
        borderColor: [
          'rgba(34, 197, 94, 1)',
          'rgba(59, 130, 246, 1)',
          'rgba(251, 191, 36, 1)',
          'rgba(239, 68, 68, 1)',
          'rgba(139, 92, 246, 1)',
        ],
        borderWidth: 2,
      },
    ],
  };

  // Statistics
  const stats = {
    totalStocks: recommendations.length,
    avgTechnicalScore: recommendations.length > 0 ?
      (recommendations.reduce((sum, rec) => sum + rec.technical_score, 0) / recommendations.length).toFixed(2) : '0',
    avgFundamentalScore: recommendations.length > 0 ?
      (recommendations.reduce((sum, rec) => sum + rec.fundamental_score, 0) / recommendations.length).toFixed(2) : '0',
    strongBuyCount: recommendations.filter(rec => rec.recommendation_strength === 'Strong Buy').length,
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8 md:space-y-12">
      {/* Hero Section */}
      <div className="text-center">
        <div className="inline-flex items-center justify-center p-3 bg-blue-100 dark:bg-blue-900/30 rounded-full mb-6">
          <ChartBarIcon className="h-12 w-12 text-blue-600 dark:text-blue-400" />
        </div>
        <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-gray-900 dark:text-gray-100 mb-4">
          Stock Advice Dashboard
        </h1>
        <p className="text-lg sm:text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto px-4">
          AI-powered stock analysis and recommendations for smart investing.
          Advanced algorithms analyze market trends, technical indicators, and fundamental data
          to provide actionable investment insights.
        </p>
      </div>

      {/* Charts Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4 sm:p-6 lg:p-8 border border-gray-200 dark:border-gray-700">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6">
          <h3 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-4 sm:mb-0">
            Top Stocks Analysis
          </h3>

          {/* Top N Dropdown */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Show Top:
            </label>
            <select
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {Array.from({ length: 20 }, (_, i) => i + 1).map(num => (
                <option key={num} value={num}>{num}</option>
              ))}
            </select>
            <span className="text-sm text-gray-500 dark:text-gray-400">stocks</span>
          </div>
        </div>

        {/* 2x2 Grid of Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Chart 1: Top Stocks by Backtest Returns */}
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 text-center">
              Best Backtest Returns (CAGR %)
            </h4>
            <div className="h-64">
              <Bar
                data={backtestReturnsData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      display: false
                    }
                  },
                  scales: {
                    y: {
                      beginAtZero: true
                    }
                  }
                }}
              />
            </div>
          </div>

          {/* Chart 2: Combined Scores */}
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 text-center">
              Combined Scores
            </h4>
            <div className="h-64">
              <Bar
                data={profitPercentageData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      display: false
                    }
                  },
                  scales: {
                    y: {
                      beginAtZero: true
                    }
                  }
                }}
              />
            </div>
          </div>

          {/* Chart 3: Technical vs Fundamental Scores */}
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 text-center">
              Technical vs Fundamental Scores
            </h4>
            <div className="h-64">
              <Bar
                data={scoresComparisonData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  scales: {
                    y: {
                      beginAtZero: true
                    }
                  }
                }}
              />
            </div>
          </div>

          {/* Chart 4: Recommendation Strength Distribution */}
          <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 text-center">
              Recommendation Distribution
            </h4>
            <div className="h-64 flex items-center justify-center">
              <div className="w-48 h-48">
                <Doughnut
                  data={doughnutData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom' as const,
                        labels: {
                          boxWidth: 12,
                          padding: 8,
                          font: {
                            size: 10
                          }
                        }
                      }
                    }
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* App Information Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
        {/* What We Do */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 sm:p-8 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center mb-6">
            <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <SparklesIcon className="h-8 w-8 text-blue-600 dark:text-blue-400" />
            </div>
            <h3 className="text-xl sm:text-2xl font-semibold text-gray-900 dark:text-gray-100 ml-4">What We Do</h3>
          </div>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
            Our platform combines machine learning algorithms with traditional financial analysis
            to evaluate stocks across multiple dimensions including technical patterns,
            fundamental metrics, and market sentiment.
          </p>
        </div>

        {/* How It Works */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center mb-6">
            <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <CogIcon className="h-8 w-8 text-green-600 dark:text-green-400" />
            </div>
            <h3 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 ml-4">How It Works</h3>
          </div>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
            Generate comprehensive analysis by configuring parameters, analyzing market data,
            and receiving scored recommendations. View detailed insights including risk assessments,
            price targets, and technical indicators.
          </p>
        </div>

        {/* Key Benefits */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center mb-6">
            <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <ShieldCheckIcon className="h-8 w-8 text-purple-600 dark:text-purple-400" />
            </div>
            <h3 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 ml-4">Key Benefits</h3>
          </div>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
            Data-driven decisions backed by comprehensive analysis, real-time market data integration,
            and systematic evaluation processes that remove emotional bias from investment choices.
          </p>
        </div>
      </div>



    </div>
  );
}
