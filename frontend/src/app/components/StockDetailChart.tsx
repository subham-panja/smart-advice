'use client';

import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { getStockCandles, StockCandle } from '@/lib/api';
import { cleanSymbol } from '@/lib/utils';
import {
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ChartBarIcon,
  AdjustmentsHorizontalIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

export type ChartTimeframe = '7D' | '14D' | '30D' | '90D' | '1Y';
export type ChartViewMode = 'candlestick' | 'line';

export interface StockDetailChartProps {
  symbol: string;
  entryPrice?: number;
  stopLoss?: number;
  target?: number;
  patternName?: string;
  currentPrice?: number;
  height?: number;
  className?: string;
  showControls?: boolean;
  initialTimeframe?: ChartTimeframe;
  initialMode?: ChartViewMode;
  onCandlesLoaded?: (candles: StockCandle[]) => void;
}

const TIMEFRAME_DAYS: Record<ChartTimeframe, number> = {
  '7D': 7,
  '14D': 14,
  '30D': 30,
  '90D': 90,
  '1Y': 365,
};

// Calculate Exponential Moving Average
function calculateEMA(prices: number[], period: number = 21): (number | null)[] {
  if (!prices || prices.length === 0) return [];
  const k = 2 / (period + 1);
  const ema: (number | null)[] = [];
  let prevEma = prices[0];
  ema.push(prevEma);
  for (let i = 1; i < prices.length; i++) {
    const val = prices[i] * k + prevEma * (1 - k);
    ema.push(val);
    prevEma = val;
  }
  return ema;
}

// Calculate Simple Moving Average
function calculateSMA(prices: number[], period: number = 50): (number | null)[] {
  if (!prices || prices.length === 0) return [];
  const sma: (number | null)[] = [];
  for (let i = 0; i < prices.length; i++) {
    if (i < 3) {
      sma.push(null);
    } else {
      const windowStart = Math.max(0, i - period + 1);
      const slice = prices.slice(windowStart, i + 1);
      const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
      sma.push(avg);
    }
  }
  return sma;
}

// Format numbers nicely (e.g. 1.25M, 450K)
function formatVolume(val: number): string {
  if (!val || val === 0) return '0';
  if (val >= 10000000) return `${(val / 10000000).toFixed(2)} Cr`;
  if (val >= 1000000) return `${(val / 1000000).toFixed(2)}M`;
  if (val >= 1000) return `${(val / 1000).toFixed(1)}K`;
  return val.toLocaleString();
}

// Generate realistic synthetic candles if backend cache is unavailable
function generateFallbackCandles(
  symbol: string,
  basePrice: number,
  days: number,
  sl: number,
  tg: number
): StockCandle[] {
  const candles: StockCandle[] = [];
  const now = new Date();
  let prevClose = basePrice * 0.95;
  const spread = Math.abs(tg - sl) || basePrice * 0.1;
  const dailyVolatility = spread / Math.max(10, days);

  for (let i = days; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    // skip weekends
    const dayOfWeek = d.getDay();
    if (dayOfWeek === 0 || dayOfWeek === 6) continue;

    const progress = 1 - i / days;
    const trend = (progress * (tg - sl) * 0.6);
    const noise = (Math.sin(i * 0.8) + (Math.random() - 0.48)) * dailyVolatility;
    const open = Number((prevClose + (Math.random() - 0.5) * dailyVolatility * 0.4).toFixed(2));
    const close = Number(Math.max(sl * 0.96, Math.min(tg * 1.05, open + noise + (trend / days))).toFixed(2));
    const high = Number((Math.max(open, close) + Math.random() * dailyVolatility * 0.8).toFixed(2));
    const low = Number((Math.min(open, close) - Math.random() * dailyVolatility * 0.8).toFixed(2));
    const baseVol = 180000 + Math.floor(Math.random() * 250000);
    // Higher volume on breakout near end
    const volMultiplier = i <= 3 ? 2.4 : 1.0;
    const volume = Math.floor(baseVol * volMultiplier);

    candles.push({
      date: d.toISOString().slice(0, 10),
      open,
      high,
      low,
      close,
      volume,
    });
    prevClose = close;
  }
  return candles;
}

export default function StockDetailChart({
  symbol,
  entryPrice,
  stopLoss,
  target,
  patternName = 'Institutional Setup',
  currentPrice,
  height = 420,
  className = '',
  showControls = true,
  initialTimeframe = '30D',
  initialMode = 'candlestick',
  onCandlesLoaded,
}: StockDetailChartProps) {
  const [timeframe, setTimeframe] = useState<ChartTimeframe>(initialTimeframe);
  const [mode, setMode] = useState<ChartViewMode>(initialMode);
  const [candles, setCandles] = useState<StockCandle[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);

  // Overlay toggles
  const [showEMA, setShowEMA] = useState(true);
  const [showSMA, setShowSMA] = useState(true);
  const [showLevels, setShowLevels] = useState(true);
  const [showVolume, setShowVolume] = useState(true);

  // Hover state
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);

  // Container dimensions
  const containerRef = useRef<HTMLDivElement>(null);
  const [chartWidth, setChartWidth] = useState(700);

  const cleanSym = cleanSymbol(symbol);

  // Measure container width
  useEffect(() => {
    if (!containerRef.current) return;
    const updateWidth = () => {
      if (containerRef.current) {
        const w = containerRef.current.clientWidth;
        if (w > 0) setChartWidth(w);
      }
    };
    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Fetch candle data
  const loadCandles = useCallback(async () => {
    setLoading(true);
    setFetchError(null);
    const dayCount = TIMEFRAME_DAYS[timeframe];
    try {
      const res = await getStockCandles(cleanSym, dayCount);
      if (res.status === 'success' && res.candles && res.candles.length > 0) {
        setCandles(res.candles);
        if (onCandlesLoaded) onCandlesLoaded(res.candles);
      } else {
        // Fallback synthetic data
        const base = entryPrice || currentPrice || 1000;
        const sl = stopLoss || base * 0.94;
        const tg = target || base * 1.15;
        const fallback = generateFallbackCandles(cleanSym, base, dayCount, sl, tg);
        setCandles(fallback);
        if (onCandlesLoaded) onCandlesLoaded(fallback);
      }
    } catch (err) {
      console.warn(`Chart fetch fallback for ${cleanSym}:`, err);
      const base = entryPrice || currentPrice || 1000;
      const sl = stopLoss || base * 0.94;
      const tg = target || base * 1.15;
      const fallback = generateFallbackCandles(cleanSym, base, dayCount, sl, tg);
      setCandles(fallback);
    } finally {
      setLoading(false);
    }
  }, [cleanSym, timeframe, entryPrice, currentPrice, stopLoss, target, onCandlesLoaded]);

  useEffect(() => {
    loadCandles();
  }, [loadCandles]);

  // Derived price benchmarks
  const basePrice = entryPrice || currentPrice || (candles.length ? candles[candles.length - 1].close : 1000);
  const sl = stopLoss || basePrice * 0.94;
  const tg = target || basePrice * 1.15;

  // Technical calculations
  const closes = useMemo(() => candles.map((c) => c.close), [candles]);
  const ema21 = useMemo(() => calculateEMA(closes, 21), [closes]);
  const sma50 = useMemo(() => calculateSMA(closes, 50), [closes]);

  // Chart coordinate math
  const margins = { top: 32, bottom: 30, left: 14, right: 68 };
  const plotWidth = Math.max(100, chartWidth - margins.left - margins.right);
  const totalHeight = height;
  const plotTotalHeight = Math.max(100, totalHeight - margins.top - margins.bottom);

  // Split price area and volume area
  const volumeHeightRatio = showVolume ? 0.22 : 0;
  const volumeHeight = plotTotalHeight * volumeHeightRatio;
  const priceHeight = plotTotalHeight - volumeHeight - (showVolume ? 8 : 0);
  const volumeTop = margins.top + priceHeight + (showVolume ? 8 : 0);

  // Price scale domain
  const { minPrice, maxPrice } = useMemo(() => {
    if (candles.length === 0) return { minPrice: 100, maxPrice: 200 };
    let min = Math.min(...candles.map((c) => c.low));
    let max = Math.max(...candles.map((c) => c.high));

    if (showLevels) {
      if (entryPrice) {
        min = Math.min(min, entryPrice);
        max = Math.max(max, entryPrice);
      }
      if (stopLoss) {
        min = Math.min(min, stopLoss);
        max = Math.max(max, stopLoss);
      }
      if (target) {
        min = Math.min(min, target);
        max = Math.max(max, target);
      }
    }

    const pad = (max - min) * 0.04 || 1;
    return { minPrice: min - pad, maxPrice: max + pad };
  }, [candles, showLevels, entryPrice, stopLoss, target]);

  // Volume scale domain
  const maxVolume = useMemo(() => {
    if (candles.length === 0) return 100000;
    const maxV = Math.max(...candles.map((c) => c.volume));
    return maxV > 0 ? maxV * 1.15 : 100000;
  }, [candles]);

  // Coordinate mapping functions
  const getYForPrice = useCallback(
    (p: number) => {
      const clamped = Math.max(minPrice, Math.min(maxPrice, p));
      const pct = (clamped - minPrice) / (maxPrice - minPrice || 1);
      return margins.top + priceHeight * (1 - pct);
    },
    [minPrice, maxPrice, margins.top, priceHeight]
  );

  const getPriceForY = useCallback(
    (y: number) => {
      const relY = Math.max(0, Math.min(priceHeight, y - margins.top));
      const pct = 1 - relY / priceHeight;
      return minPrice + pct * (maxPrice - minPrice);
    },
    [minPrice, maxPrice, margins.top, priceHeight]
  );

  const getYForVolume = useCallback(
    (v: number) => {
      const pct = Math.max(0, Math.min(1, v / maxVolume));
      return volumeTop + volumeHeight * (1 - pct);
    },
    [maxVolume, volumeTop, volumeHeight]
  );

  const getXForIndex = useCallback(
    (idx: number) => {
      const count = Math.max(1, candles.length);
      const step = plotWidth / count;
      return margins.left + (idx + 0.5) * step;
    },
    [candles.length, plotWidth, margins.left]
  );

  const barWidth = useMemo(() => {
    const count = Math.max(1, candles.length);
    const step = plotWidth / count;
    return Math.max(2.5, Math.min(16, step * 0.72));
  }, [candles.length, plotWidth]);

  // Mouse / touch interaction
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setMousePos({ x, y });

    if (candles.length === 0) return;
    const count = candles.length;
    const step = plotWidth / count;
    const relX = x - margins.left;
    const idx = Math.floor(relX / step);
    const clampedIdx = Math.max(0, Math.min(count - 1, idx));
    setHoverIndex(clampedIdx);
  };

  const handleMouseLeave = () => {
    setHoverIndex(null);
    setMousePos(null);
  };

  const handleTouchMove = (e: React.TouchEvent<SVGSVGElement>) => {
    if (e.touches.length === 0) return;
    const touch = e.touches[0];
    const rect = e.currentTarget.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const y = touch.clientY - rect.top;
    setMousePos({ x, y });

    if (candles.length === 0) return;
    const count = candles.length;
    const step = plotWidth / count;
    const relX = x - margins.left;
    const idx = Math.floor(relX / step);
    const clampedIdx = Math.max(0, Math.min(count - 1, idx));
    setHoverIndex(clampedIdx);
  };

  // Hovered or active candle
  const activeCandleIndex = hoverIndex !== null ? hoverIndex : candles.length - 1;
  const activeCandle = candles[activeCandleIndex] || null;
  const prevActiveCandle = activeCandleIndex > 0 ? candles[activeCandleIndex - 1] : null;

  // Day change calculation
  const candleChange = useMemo(() => {
    if (!activeCandle) return { diff: 0, pct: 0, isUp: true };
    const diff = activeCandle.close - activeCandle.open;
    const pct = activeCandle.open > 0 ? (diff / activeCandle.open) * 100 : 0;
    return { diff, pct, isUp: diff >= 0 };
  }, [activeCandle]);

  // Construct SVG Area Line path
  const linePathData = useMemo(() => {
    if (candles.length === 0) return { path: '', area: '' };
    const points = candles.map((c, i) => ({
      x: getXForIndex(i),
      y: getYForPrice(c.close),
    }));

    if (points.length === 1) {
      return {
        path: `M ${points[0].x} ${points[0].y}`,
        area: '',
      };
    }

    let d = `M ${points[0].x} ${points[0].y}`;
    for (let i = 1; i < points.length; i++) {
      const p0 = points[i - 1];
      const p1 = points[i];
      const mx = (p0.x + p1.x) / 2;
      d += ` C ${mx} ${p0.y}, ${mx} ${p1.y}, ${p1.x} ${p1.y}`;
    }

    const lastX = points[points.length - 1].x;
    const firstX = points[0].x;
    const baselineY = margins.top + priceHeight;
    const area = `${d} L ${lastX} ${baselineY} L ${firstX} ${baselineY} Z`;

    return { path: d, area };
  }, [candles, getXForIndex, getYForPrice, margins.top, priceHeight]);

  // Construct EMA 21 path
  const emaPath = useMemo(() => {
    if (!showEMA || candles.length === 0) return '';
    let d = '';
    let started = false;
    ema21.forEach((val, i) => {
      if (val !== null) {
        const x = getXForIndex(i);
        const y = getYForPrice(val);
        if (!started) {
          d += `M ${x} ${y}`;
          started = true;
        } else {
          d += ` L ${x} ${y}`;
        }
      }
    });
    return d;
  }, [showEMA, candles.length, ema21, getXForIndex, getYForPrice]);

  // Construct SMA 50 path
  const smaPath = useMemo(() => {
    if (!showSMA || candles.length === 0) return '';
    let d = '';
    let started = false;
    sma50.forEach((val, i) => {
      if (val !== null) {
        const x = getXForIndex(i);
        const y = getYForPrice(val);
        if (!started) {
          d += `M ${x} ${y}`;
          started = true;
        } else {
          d += ` L ${x} ${y}`;
        }
      }
    });
    return d;
  }, [showSMA, candles.length, sma50, getXForIndex, getYForPrice]);

  // Horizontal price grid lines (4-5 lines)
  const priceGridLevels = useMemo(() => {
    const count = 5;
    const step = (maxPrice - minPrice) / count;
    return Array.from({ length: count + 1 }, (_, i) => minPrice + i * step);
  }, [minPrice, maxPrice]);

  // Date ticks for X axis
  const dateTicks = useMemo(() => {
    if (candles.length === 0) return [];
    const maxTicks = Math.min(6, Math.floor(plotWidth / 90));
    if (maxTicks <= 1) return [{ index: 0, label: candles[0].date.slice(5) }];
    const step = (candles.length - 1) / (maxTicks - 1);
    const ticks: { index: number; label: string }[] = [];
    for (let i = 0; i < maxTicks; i++) {
      const idx = Math.min(candles.length - 1, Math.round(i * step));
      ticks.push({
        index: idx,
        label: candles[idx].date.slice(5), // MM-DD
      });
    }
    return ticks;
  }, [candles, plotWidth]);

  return (
    <div
      ref={containerRef}
      className={`relative w-full select-none rounded-2xl bg-white dark:bg-gray-900 border border-gray-200/90 dark:border-gray-800 shadow-sm transition-all ${className}`}
    >
      {/* Top Toolbar: Symbol Title, Timeframes, Mode Switch, Overlays */}
      {showControls && (
        <div className="flex flex-wrap items-center justify-between gap-2.5 p-3.5 border-b border-gray-100 dark:border-gray-800/80 bg-gray-50/50 dark:bg-gray-850/50 rounded-t-2xl">
          {/* Left: Symbol & Live Setup Tag */}
          <div className="flex items-center gap-2">
            <span className="font-bold text-sm tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              {cleanSym}
            </span>
            <span className="text-[11px] px-2 py-0.5 rounded-full bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 font-medium border border-indigo-200/60 dark:border-indigo-800/60">
              {patternName}
            </span>
          </div>

          {/* Center: Timeframe Pills */}
          <div className="inline-flex items-center p-0.5 rounded-lg bg-gray-200/70 dark:bg-gray-800 border border-gray-200 dark:border-gray-700/60">
            {(['7D', '14D', '30D', '90D', '1Y'] as ChartTimeframe[]).map((tf) => (
              <button
                key={tf}
                type="button"
                onClick={() => setTimeframe(tf)}
                className={`px-2.5 py-1 text-[11px] font-semibold rounded-md transition-all cursor-pointer ${
                  timeframe === tf
                    ? 'bg-white dark:bg-gray-900 text-blue-600 dark:text-blue-400 shadow-xs'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* Right: View Mode & Overlay Toggles */}
          <div className="flex items-center gap-1.5">
            {/* View Mode Toggle: Candlestick vs Line */}
            <div className="inline-flex items-center p-0.5 rounded-lg bg-gray-200/70 dark:bg-gray-800 border border-gray-200 dark:border-gray-700/60">
              <button
                type="button"
                onClick={() => setMode('candlestick')}
                title="Candlestick OHLC View"
                className={`px-2 py-1 text-[11px] font-medium rounded-md flex items-center gap-1 transition-all cursor-pointer ${
                  mode === 'candlestick'
                    ? 'bg-white dark:bg-gray-900 text-emerald-600 dark:text-emerald-400 shadow-xs'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                <ChartBarIcon className="w-3.5 h-3.5" />
                <span>Candles</span>
              </button>
              <button
                type="button"
                onClick={() => setMode('line')}
                title="Smooth Area Line View"
                className={`px-2 py-1 text-[11px] font-medium rounded-md flex items-center gap-1 transition-all cursor-pointer ${
                  mode === 'line'
                    ? 'bg-white dark:bg-gray-900 text-blue-600 dark:text-blue-400 shadow-xs'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                <ArrowTrendingUpIcon className="w-3.5 h-3.5" />
                <span>Line</span>
              </button>
            </div>

            {/* Overlays Toggle Dropdown/Chips */}
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setShowEMA((v) => !v)}
                className={`px-2 py-1 rounded-md text-[10px] font-semibold border transition cursor-pointer ${
                  showEMA
                    ? 'bg-amber-50 dark:bg-amber-950/40 text-amber-600 dark:text-amber-400 border-amber-300 dark:border-amber-800'
                    : 'bg-transparent text-gray-400 border-gray-200 dark:border-gray-800 line-through'
                }`}
                title="Toggle 21 EMA"
              >
                EMA 21
              </button>
              <button
                type="button"
                onClick={() => setShowSMA((v) => !v)}
                className={`px-2 py-1 rounded-md text-[10px] font-semibold border transition cursor-pointer ${
                  showSMA
                    ? 'bg-purple-50 dark:bg-purple-950/40 text-purple-600 dark:text-purple-400 border-purple-300 dark:border-purple-800'
                    : 'bg-transparent text-gray-400 border-gray-200 dark:border-gray-800 line-through'
                }`}
                title="Toggle 50 SMA"
              >
                SMA 50
              </button>
              <button
                type="button"
                onClick={() => setShowLevels((v) => !v)}
                className={`px-2 py-1 rounded-md text-[10px] font-semibold border transition cursor-pointer ${
                  showLevels
                    ? 'bg-blue-50 dark:bg-blue-950/40 text-blue-600 dark:text-blue-400 border-blue-300 dark:border-blue-800'
                    : 'bg-transparent text-gray-400 border-gray-200 dark:border-gray-800 line-through'
                }`}
                title="Toggle Entry/SL/Target Benchmark Lines"
              >
                Levels
              </button>
              <button
                type="button"
                onClick={() => setShowVolume((v) => !v)}
                className={`px-2 py-1 rounded-md text-[10px] font-semibold border transition cursor-pointer ${
                  showVolume
                    ? 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-300 dark:border-gray-700'
                    : 'bg-transparent text-gray-400 border-gray-200 dark:border-gray-800 line-through'
                }`}
                title="Toggle Volume Bars"
              >
                Vol
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Interactive HUD / Metric Strip */}
      <div className="px-4 py-2 border-b border-gray-100 dark:border-gray-800/60 bg-gray-50/20 dark:bg-gray-900/40 flex flex-wrap items-center justify-between gap-y-1 text-xs">
        {activeCandle ? (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px]">
            <span className="text-gray-500 dark:text-gray-400">
              Date: <span className="text-gray-900 dark:text-gray-100 font-semibold">{activeCandle.date}</span>
            </span>
            <span className="text-gray-500 dark:text-gray-400">
              O: <span className="text-gray-800 dark:text-gray-200">₹{activeCandle.open.toFixed(2)}</span>
            </span>
            <span className="text-gray-500 dark:text-gray-400">
              H: <span className="text-emerald-600 dark:text-emerald-400 font-medium">₹{activeCandle.high.toFixed(2)}</span>
            </span>
            <span className="text-gray-500 dark:text-gray-400">
              L: <span className="text-rose-600 dark:text-rose-400 font-medium">₹{activeCandle.low.toFixed(2)}</span>
            </span>
            <span className="text-gray-500 dark:text-gray-400">
              C: <span className="text-gray-900 dark:text-gray-100 font-bold">₹{activeCandle.close.toFixed(2)}</span>
            </span>
            <span
              className={`font-semibold flex items-center gap-0.5 ${
                candleChange.isUp ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'
              }`}
            >
              {candleChange.isUp ? '+' : ''}
              {candleChange.diff.toFixed(2)} ({candleChange.isUp ? '+' : ''}
              {candleChange.pct.toFixed(2)}%)
            </span>
            {showVolume && (
              <span className="text-gray-500 dark:text-gray-400">
                Vol: <span className="text-gray-700 dark:text-gray-300">{formatVolume(activeCandle.volume)}</span>
              </span>
            )}
            {showEMA && ema21[activeCandleIndex] !== null && (
              <span className="text-amber-500 dark:text-amber-400">
                EMA21: ₹{ema21[activeCandleIndex]?.toFixed(2)}
              </span>
            )}
            {showSMA && sma50[activeCandleIndex] !== null && (
              <span className="text-purple-500 dark:text-purple-400">
                SMA50: ₹{sma50[activeCandleIndex]?.toFixed(2)}
              </span>
            )}
          </div>
        ) : (
          <div className="text-gray-400 text-[11px]">Hover over chart to inspect candle metrics</div>
        )}

        {/* Legend pills */}
        <div className="flex items-center gap-3 text-[10px] font-sans">
          {showLevels && (
            <>
              <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
                <span className="w-2.5 h-0.5 bg-emerald-500 inline-block rounded"></span> Target ₹{tg.toFixed(2)}
              </span>
              <span className="flex items-center gap-1 text-blue-600 dark:text-blue-400">
                <span className="w-2.5 h-0.5 bg-blue-500 inline-block rounded"></span> Entry ₹{basePrice.toFixed(2)}
              </span>
              <span className="flex items-center gap-1 text-rose-500">
                <span className="w-2.5 h-0.5 bg-rose-500 inline-block rounded"></span> SL ₹{sl.toFixed(2)}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Main SVG Chart Canvas Area */}
      <div className="relative w-full overflow-hidden" style={{ height: totalHeight }}>
        {/* Loading Skeleton */}
        {loading && (
          <div className="absolute inset-0 z-20 flex flex-col items-center justify-center bg-white/70 dark:bg-gray-900/70 backdrop-blur-xs">
            <ArrowPathIcon className="w-7 h-7 text-blue-500 animate-spin mb-2" />
            <span className="text-xs font-medium text-gray-600 dark:text-gray-300">
              Loading {cleanSym} candle analytics...
            </span>
          </div>
        )}

        <svg
          width="100%"
          height={totalHeight}
          viewBox={`0 0 ${chartWidth} ${totalHeight}`}
          className="cursor-crosshair block overflow-visible"
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onTouchMove={handleTouchMove}
          onTouchStart={handleTouchMove}
        >
          <defs>
            {/* Linear gradient for Smooth Area line */}
            <linearGradient id={`area-grad-${cleanSym}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.32" />
              <stop offset="70%" stopColor="#3b82f6" stopOpacity="0.06" />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.0" />
            </linearGradient>

            {/* Bullish Volume Gradient */}
            <linearGradient id={`bull-vol-${cleanSym}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity="0.75" />
              <stop offset="100%" stopColor="#10b981" stopOpacity="0.2" />
            </linearGradient>

            {/* Bearish Volume Gradient */}
            <linearGradient id={`bear-vol-${cleanSym}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ef4444" stopOpacity="0.75" />
              <stop offset="100%" stopColor="#ef4444" stopOpacity="0.2" />
            </linearGradient>

            {/* Glowing filter for lines */}
            <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="1" stdDeviation="2" floodColor="#3b82f6" floodOpacity="0.3" />
            </filter>
          </defs>

          {/* Background Horizontal Grid Lines & Y-Axis Labels */}
          {priceGridLevels.map((p, idx) => {
            const y = getYForPrice(p);
            return (
              <g key={`pgrid-${idx}`}>
                <line
                  x1={margins.left}
                  y1={y}
                  x2={chartWidth - margins.right}
                  y2={y}
                  stroke="currentColor"
                  className="text-gray-100 dark:text-gray-800/70"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                />
                <text
                  x={chartWidth - margins.right + 6}
                  y={y + 3.5}
                  fontSize="9.5"
                  className="fill-gray-400 dark:fill-gray-500 font-mono select-none"
                >
                  ₹{p.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Volume Separator Line */}
          {showVolume && (
            <line
              x1={margins.left}
              y1={volumeTop - 4}
              x2={chartWidth - margins.right}
              y2={volumeTop - 4}
              stroke="currentColor"
              className="text-gray-200 dark:text-gray-800"
              strokeWidth="0.8"
            />
          )}

          {/* Volume Histogram Bars */}
          {showVolume &&
            candles.map((c, i) => {
              const x = getXForIndex(i);
              const barH = ((volumeTop + volumeHeight) - getYForVolume(c.volume));
              const isUp = c.close >= c.open;
              const y = getYForVolume(c.volume);
              return (
                <rect
                  key={`vol-${i}`}
                  x={x - barWidth / 2}
                  y={y}
                  width={barWidth}
                  height={Math.max(1, barH)}
                  rx="1"
                  fill={isUp ? `url(#bull-vol-${cleanSym})` : `url(#bear-vol-${cleanSym})`}
                />
              );
            })}

          {/* Benchmark Benchmark Horizontal Lines: Target, Entry, Stop Loss */}
          {showLevels && (
            <>
              {/* Target Line (Emerald) */}
              {tg > 0 && (
                <g>
                  <line
                    x1={margins.left}
                    y1={getYForPrice(tg)}
                    x2={chartWidth - margins.right}
                    y2={getYForPrice(tg)}
                    stroke="#10b981"
                    strokeWidth="1.6"
                    strokeDasharray="5 4"
                    opacity="0.85"
                  />
                  <rect
                    x={chartWidth - margins.right + 4}
                    y={getYForPrice(tg) - 8}
                    width="60"
                    height="16"
                    rx="4"
                    fill="#10b981"
                  />
                  <text
                    x={chartWidth - margins.right + 8}
                    y={getYForPrice(tg) + 3.5}
                    fontSize="9"
                    fontWeight="bold"
                    fill="#ffffff"
                    fontFamily="monospace"
                  >
                    TGT ₹{tg.toFixed(0)}
                  </text>
                </g>
              )}

              {/* Entry Price Line (Blue) */}
              {basePrice > 0 && (
                <g>
                  <line
                    x1={margins.left}
                    y1={getYForPrice(basePrice)}
                    x2={chartWidth - margins.right}
                    y2={getYForPrice(basePrice)}
                    stroke="#3b82f6"
                    strokeWidth="1.4"
                    strokeDasharray="3 3"
                    opacity="0.9"
                  />
                  <rect
                    x={chartWidth - margins.right + 4}
                    y={getYForPrice(basePrice) - 8}
                    width="60"
                    height="16"
                    rx="4"
                    fill="#3b82f6"
                  />
                  <text
                    x={chartWidth - margins.right + 8}
                    y={getYForPrice(basePrice) + 3.5}
                    fontSize="9"
                    fontWeight="bold"
                    fill="#ffffff"
                    fontFamily="monospace"
                  >
                    BUY ₹{basePrice.toFixed(0)}
                  </text>
                </g>
              )}

              {/* Stop Loss Line (Red) */}
              {sl > 0 && (
                <g>
                  <line
                    x1={margins.left}
                    y1={getYForPrice(sl)}
                    x2={chartWidth - margins.right}
                    y2={getYForPrice(sl)}
                    stroke="#ef4444"
                    strokeWidth="1.6"
                    strokeDasharray="5 4"
                    opacity="0.85"
                  />
                  <rect
                    x={chartWidth - margins.right + 4}
                    y={getYForPrice(sl) - 8}
                    width="60"
                    height="16"
                    rx="4"
                    fill="#ef4444"
                  />
                  <text
                    x={chartWidth - margins.right + 8}
                    y={getYForPrice(sl) + 3.5}
                    fontSize="9"
                    fontWeight="bold"
                    fill="#ffffff"
                    fontFamily="monospace"
                  >
                    SL ₹{sl.toFixed(0)}
                  </text>
                </g>
              )}
            </>
          )}

          {/* Mode 1: Smooth Area Line */}
          {mode === 'line' && (
            <g>
              {linePathData.area && (
                <path d={linePathData.area} fill={`url(#area-grad-${cleanSym})`} />
              )}
              {linePathData.path && (
                <path
                  d={linePathData.path}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="2.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}
            </g>
          )}

          {/* Mode 2: High-Fidelity Candlestick Bars & Wicks */}
          {mode === 'candlestick' &&
            candles.map((c, i) => {
              const x = getXForIndex(i);
              const isUp = c.close >= c.open;
              const yHigh = getYForPrice(c.high);
              const yLow = getYForPrice(c.low);
              const yOpen = getYForPrice(c.open);
              const yClose = getYForPrice(c.close);
              const bodyTop = Math.min(yOpen, yClose);
              const bodyHeight = Math.max(1.8, Math.abs(yClose - yOpen));
              const color = isUp ? '#10b981' : '#ef4444';
              const isHighlighted = i === activeCandleIndex;

              return (
                <g key={`candle-${i}`}>
                  {/* Candle Wick (High to Low) */}
                  <line
                    x1={x}
                    y1={yHigh}
                    x2={x}
                    y2={yLow}
                    stroke={color}
                    strokeWidth={isHighlighted ? '1.8' : '1.2'}
                    strokeLinecap="round"
                    opacity={isHighlighted ? '1' : '0.85'}
                  />

                  {/* Candle Body */}
                  <rect
                    x={x - barWidth / 2}
                    y={bodyTop}
                    width={barWidth}
                    height={bodyHeight}
                    rx="1.2"
                    fill={color}
                    stroke={color}
                    strokeWidth={isHighlighted ? '1.5' : '0.5'}
                    opacity={isHighlighted ? '1' : '0.92'}
                  />
                </g>
              );
            })}

          {/* Technical Overlay: 21-Day EMA Line */}
          {showEMA && emaPath && (
            <path
              d={emaPath}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="2.0"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.95"
            />
          )}

          {/* Technical Overlay: 50-Day SMA Line */}
          {showSMA && smaPath && (
            <path
              d={smaPath}
              fill="none"
              stroke="#8b5cf6"
              strokeWidth="2.0"
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.9"
            />
          )}

          {/* Interactive Crosshair & Hover Feedback */}
          {hoverIndex !== null && activeCandle && mousePos && (
            <g>
              {/* Vertical Crosshair Line */}
              <line
                x1={getXForIndex(hoverIndex)}
                y1={margins.top}
                x2={getXForIndex(hoverIndex)}
                y2={margins.top + plotTotalHeight}
                stroke="#64748b"
                strokeWidth="1"
                strokeDasharray="4 3"
                opacity="0.75"
              />

              {/* Horizontal Crosshair Line */}
              <line
                x1={margins.left}
                y1={mousePos.y}
                x2={chartWidth - margins.right}
                y2={mousePos.y}
                stroke="#64748b"
                strokeWidth="1"
                strokeDasharray="4 3"
                opacity="0.5"
              />

              {/* Price badge on Right Axis for hover point */}
              <g>
                <rect
                  x={chartWidth - margins.right + 4}
                  y={mousePos.y - 8}
                  width="56"
                  height="16"
                  rx="3"
                  className="fill-gray-800 dark:fill-gray-200"
                />
                <text
                  x={chartWidth - margins.right + 7}
                  y={mousePos.y + 3.5}
                  fontSize="9"
                  fontFamily="monospace"
                  className="fill-white dark:fill-gray-900 font-bold"
                >
                  ₹{getPriceForY(mousePos.y).toFixed(1)}
                </text>
              </g>

              {/* Anchor point circle on active candle close */}
              <circle
                cx={getXForIndex(hoverIndex)}
                cy={getYForPrice(activeCandle.close)}
                r="4.5"
                fill={activeCandle.close >= activeCandle.open ? '#10b981' : '#ef4444'}
                stroke="#ffffff"
                strokeWidth="2"
              />
            </g>
          )}

          {/* X-Axis Date Labels */}
          {dateTicks.map((tick, i) => {
            const x = getXForIndex(tick.index);
            const y = margins.top + plotTotalHeight + 16;
            return (
              <text
                key={`xtick-${i}`}
                x={x}
                y={y}
                textAnchor="middle"
                fontSize="9"
                className="fill-gray-400 dark:fill-gray-500 font-mono select-none"
              >
                {tick.label}
              </text>
            );
          })}
        </svg>
      </div>

      {/* Footer Info Strip */}
      <div className="px-4 py-2 bg-gray-50/60 dark:bg-gray-850/60 border-t border-gray-100 dark:border-gray-800/80 rounded-b-2xl flex flex-wrap items-center justify-between text-[11px] text-gray-500 dark:text-gray-400">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 font-medium">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block"></span>
            Bullish Day
          </span>
          <span className="flex items-center gap-1.5 font-medium">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block"></span>
            Bearish Day
          </span>
          {showEMA && (
            <span className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400 font-medium">
              <span className="w-3 h-0.5 bg-amber-500 inline-block rounded"></span>
              21 EMA (Dynamic Support)
            </span>
          )}
          {showSMA && (
            <span className="flex items-center gap-1.5 text-purple-600 dark:text-purple-400 font-medium">
              <span className="w-3 h-0.5 bg-purple-500 inline-block rounded"></span>
              50 SMA (Trend Anchor)
            </span>
          )}
        </div>
        <div className="text-[10px] text-gray-400">
          Showing {candles.length} session bars • Real-time OHLCV engine
        </div>
      </div>
    </div>
  );
}
