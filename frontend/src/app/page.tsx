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

  useEffect(() => {
    async function fetchData() {
      const response = await getRecommendations();
      if (response.status === 'success' && response.recommendations) {
        setRecommendations(response.recommendations);
      }
    }
    fetchData();
  }, []);

  // Chart data preparation
  const topStocks = recommendations.slice(0, 10); // Show top 10 stocks in charts
  
  const chartData = {
    labels: topStocks.map(rec => rec.symbol),
    datasets: [
      {
        label: 'Technical Score',
        data: topStocks.map(rec => rec.technical_score),
        backgroundColor: 'rgba(59, 130, 246, 0.5)',
        borderColor: 'rgba(59, 130, 246, 1)',
        borderWidth: 2,
      },
      {
        label: 'Fundamental Score',
        data: topStocks.map(rec => rec.fundamental_score),
        backgroundColor: 'rgba(16, 185, 129, 0.5)',
        borderColor: 'rgba(16, 185, 129, 1)',
        borderWidth: 2,
      },
      {
        label: 'Sentiment Score',
        data: topStocks.map(rec => rec.sentiment_score),
        backgroundColor: 'rgba(139, 92, 246, 0.5)',
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
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-6 text-center">
          Recommendations Overview
        </h3>
        <div className="my-8">
          <Bar data={chartData} />
        </div>
        <div className="my-8">
          <Line data={chartData} />
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

      {/* Current Features Section */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-8 border border-gray-200 dark:border-gray-700 transition-colors mb-8">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-6 text-center">
          Current Features
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="text-center">
            <ArrowTrendingUpIcon className="h-12 w-12 text-blue-600 dark:text-blue-400 mx-auto mb-4" />
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Technical Analysis</h4>
            <p className="text-gray-600 dark:text-gray-300">
              Advanced technical indicators and chart pattern recognition
            </p>
          </div>
          <div className="text-center">
            <ChartBarIcon className="h-12 w-12 text-green-600 dark:text-green-400 mx-auto mb-4" />
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Fundamental Analysis</h4>
            <p className="text-gray-600 dark:text-gray-300">
              Financial metrics and company performance evaluation
            </p>
          </div>
          <div className="text-center">
            <PlayIcon className="h-12 w-12 text-purple-600 dark:text-purple-400 mx-auto mb-4" />
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">Sentiment Analysis</h4>
            <p className="text-gray-600 dark:text-gray-300">
              Market sentiment and news analysis for informed decisions
            </p>
          </div>
        </div>
      </div>

      {/* Upcoming Features Section */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg shadow-md p-8 border border-blue-200 dark:border-blue-700 transition-colors">
        <div className="text-center mb-6">
          <h3 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Coming Soon: F&O Analysis
          </h3>
          <span className="inline-block bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 text-sm font-medium px-3 py-1 rounded-full">
            Next Feature
          </span>
        </div>
        <div className="max-w-2xl mx-auto">
          <p className="text-gray-600 dark:text-gray-300 text-center mb-6">
            Advanced Futures & Options analysis to help you make informed decisions in derivatives trading. 
            Get insights on option chains, volatility analysis, and risk management strategies.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Option Chain Analysis</h4>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Real-time option chain data with strike price analysis
              </p>
            </div>
            <div className="bg-white/50 dark:bg-gray-800/50 rounded-lg p-4">
              <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2">Volatility Insights</h4>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                Historical and implied volatility tracking
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Debug Section */}
      <div className="mt-12">
        <h3 className="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-6 text-center">
          System Status
        </h3>
        <ApiTest />
      </div>
    </div>
  );
}
