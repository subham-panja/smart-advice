'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import {
  CalendarDaysIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  FunnelIcon,
  ScaleIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';
import { getDailyEarnings, DailyEarningRecord } from '@/lib/api';
import { formatDecimal } from '@/lib/utils';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

type PresetFilter = '7d' | '30d' | '90d' | '1y' | 'all' | 'custom';

export default function DailyEarningsChart() {
  const [records, setRecords] = useState<DailyEarningRecord[]>([]);
  const [summary, setSummary] = useState<any>({});
  const [filter, setFilter] = useState<PresetFilter>('30d');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [loading, setLoading] = useState(true);

  const fetchEarnings = async () => {
    setLoading(true);
    let params: any = {};

    if (filter === '7d') params.days = 7;
    else if (filter === '30d') params.days = 30;
    else if (filter === '90d') params.days = 90;
    else if (filter === '1y') params.days = 252;
    else if (filter === 'custom') {
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;
    }
    // 'all' passes no params, returns all available data

    const res = await getDailyEarnings(params);
    if (res.status === 'success') {
      setRecords(res.daily_records || []);
      setSummary(res.summary || {});
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchEarnings();
  }, [filter]);

  const handleApplyCustom = () => {
    if (filter === 'custom') {
      fetchEarnings();
    } else {
      setFilter('custom');
    }
  };

  const chartData = useMemo(() => {
    const labels = records.map((r) => r.date);
    const data = records.map((r) => r.daily_pnl);
    const backgroundColors = data.map((val) =>
      val >= 0 ? 'rgba(34, 197, 94, 0.85)' : 'rgba(239, 68, 68, 0.85)'
    );
    const borderColors = data.map((val) => (val >= 0 ? '#16a34a' : '#dc2626'));

    return {
      labels,
      datasets: [
        {
          label: 'Daily Net P&L (₹)',
          data,
          backgroundColor: backgroundColors,
          borderColor: borderColors,
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    };
  }, [records]);

  const options: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1f2937',
        titleFont: { size: 12 },
        bodyFont: { size: 12 },
        padding: 10,
        callbacks: {
          label: (ctx) => {
            const raw = Number(ctx.raw || 0);
            const sign = raw >= 0 ? '+' : '';
            return `P&L: ${sign}₹${raw.toLocaleString('en-IN')}`;
          },
          afterLabel: (ctx) => {
            const rec = records[ctx.dataIndex];
            if (!rec) return '';
            return `Portfolio: ₹${rec.portfolio_value.toLocaleString('en-IN')} | Positions: ${rec.open_positions}`;
          },
        },
      },
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: {
          font: { size: 10 },
          color: '#9ca3af',
          maxTicksLimit: 12,
        },
      },
      y: {
        grid: { color: 'rgba(156, 163, 175, 0.1)' },
        ticks: {
          font: { size: 10 },
          color: '#9ca3af',
          callback: (val) => {
            const num = Number(val);
            if (Math.abs(num) >= 1e7) return `₹${(num / 1e7).toFixed(1)}Cr`;
            if (Math.abs(num) >= 1e5) return `₹${(num / 1e5).toFixed(1)}L`;
            return `₹${num}`;
          },
        },
      },
    },
  };

  const totalNet = summary.total_net_pnl ?? 0;
  const isNetPositive = totalNet >= 0;

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 p-6 space-y-4">
      {/* Top Header & Range Filters */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-gray-100 dark:border-gray-700/60 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <CalendarDaysIcon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
              Day-by-Day Earnings Report
            </h2>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium">
              Daily Realized & Mark-to-Market
            </span>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Visual bar chart of daily trading session earnings with institutional date filtering.
          </p>
        </div>

        {/* Range Buttons & Custom Picker */}
        <div className="flex flex-wrap items-center gap-2">
          {(['7d', '30d', '90d', '1y', 'all'] as PresetFilter[]).map((p) => (
            <button
              key={p}
              onClick={() => setFilter(p)}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                filter === p
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-650'
              }`}
            >
              {p === 'all' ? 'Select All' : p.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Custom Date Range Selector */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-gray-50 dark:bg-gray-850 p-3 rounded-xl border border-gray-100 dark:border-gray-750">
        <div className="flex items-center gap-2 flex-wrap text-xs">
          <FunnelIcon className="h-4 w-4 text-gray-400 shrink-0" />
          <span className="font-medium text-gray-600 dark:text-gray-300">Date Range Filter:</span>
          <input
            type="date"
            value={startDate}
            onChange={(e) => {
              setStartDate(e.target.value);
              setFilter('custom');
            }}
            className="px-2.5 py-1 text-xs rounded-lg border border-gray-300 dark:border-gray-650 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-1 focus:ring-blue-500"
          />
          <span className="text-gray-400">to</span>
          <input
            type="date"
            value={endDate}
            onChange={(e) => {
              setEndDate(e.target.value);
              setFilter('custom');
            }}
            className="px-2.5 py-1 text-xs rounded-lg border border-gray-300 dark:border-gray-650 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-1 focus:ring-blue-500"
          />
          <button
            onClick={handleApplyCustom}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-medium transition"
          >
            Apply Range
          </button>
        </div>

        {/* Quick summary stats in header bar */}
        <div className="flex items-center gap-4 text-xs">
          <div className="flex items-center gap-1.5">
            <span className="text-gray-500">Period Net P&L:</span>
            <span className={`font-bold ${isNetPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
              {isNetPositive ? '+' : ''}₹{totalNet.toLocaleString('en-IN')}
            </span>
          </div>
          <div className="hidden sm:flex items-center gap-1.5">
            <span className="text-gray-500">Win Rate:</span>
            <span className="font-semibold text-gray-900 dark:text-gray-100">
              {summary.win_rate ?? 0}% ({summary.up_days ?? 0}W / {summary.down_days ?? 0}L)
            </span>
          </div>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="relative h-64 sm:h-72 w-full pt-2">
        {loading ? (
          <div className="h-full flex items-center justify-center text-gray-400 text-sm">
            <span className="animate-spin mr-2">⟳</span> Loading daily earnings...
          </div>
        ) : records.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-400 text-sm">
            No session earnings recorded in this date range.
          </div>
        ) : (
          <Bar data={chartData} options={options} />
        )}
      </div>

      {/* KPI Cards Strip below chart */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
        <div className="bg-gray-50 dark:bg-gray-850 p-3 rounded-xl border border-gray-100 dark:border-gray-750 text-center">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider">Trading Days</div>
          <div className="text-sm font-bold text-gray-900 dark:text-gray-100">{summary.total_days ?? 0} Days</div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-850 p-3 rounded-xl border border-gray-100 dark:border-gray-750 text-center">
          <div className="text-[10px] text-green-500 uppercase tracking-wider">Best Day</div>
          <div className="text-sm font-bold text-green-600 dark:text-green-400">
            +₹{(summary.best_day_pnl ?? 0).toLocaleString('en-IN')}
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-850 p-3 rounded-xl border border-gray-100 dark:border-gray-750 text-center">
          <div className="text-[10px] text-red-500 uppercase tracking-wider">Worst Day</div>
          <div className="text-sm font-bold text-red-600 dark:text-red-400">
            ₹{(summary.worst_day_pnl ?? 0).toLocaleString('en-IN')}
          </div>
        </div>
        <div className="bg-gray-50 dark:bg-gray-850 p-3 rounded-xl border border-gray-100 dark:border-gray-750 text-center">
          <div className="text-[10px] text-blue-500 uppercase tracking-wider">Up / Down Days</div>
          <div className="text-sm font-bold text-blue-600 dark:text-blue-400">
            {summary.up_days ?? 0} Up / {summary.down_days ?? 0} Down
          </div>
        </div>
      </div>
    </div>
  );
}
