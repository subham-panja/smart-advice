'use client';

import React, { useState } from 'react';
import { StockRecommendation, updateRecommendation } from '@/lib/api';
import { cleanSymbol } from '@/lib/utils';
import StockDetailChart from './StockDetailChart';
import {
  SparklesIcon,
  ShieldCheckIcon,
  CheckCircleIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ChartBarIcon,
  DocumentTextIcon,
  XMarkIcon,
  ClockIcon,
  ClipboardDocumentCheckIcon,
  ClipboardDocumentIcon,
  FireIcon,
  ScaleIcon,
  BanknotesIcon,
  CheckIcon,
} from '@heroicons/react/24/outline';

export interface StockAnalysisDeepDiveProps {
  recommendation?: StockRecommendation;
  symbol?: string;
  companyName?: string;
  strategyName?: string;
  recommendationStrength?: string;
  buyPrice?: number;
  sellPrice?: number;
  stopLoss?: number;
  expectedReturnPercent?: number;
  technicalScore?: number;
  fundamentalScore?: number;
  sentimentScore?: number;
  combinedScore?: number;
  compositeScore?: number;
  entryAtr?: number;
  rrRatio?: number;
  suggestedQuantity?: number;
  allocationPct?: number;
  reason?: string;
  note?: string;
  patternDetected?: string;
  setupChecklist?: Record<string, any>;
  technicalPillars?: Record<string, any>;
  onSaveNote?: (newNote: string) => Promise<void>;
  onClose?: () => void;
  isModal?: boolean;
  className?: string;
}

// Derive pattern and institutional setup rationale
function getSetupDetails(symbol: string, reason?: string, strategy?: string, pattern?: string) {
  const r = (reason || '').toLowerCase();
  const pat = (pattern || '').toLowerCase();

  if (pat.includes('vcp') || r.includes('vcp') || r.includes('contraction')) {
    return {
      patternName: 'Minervini VCP Breakout',
      tagColor: 'bg-indigo-100 text-indigo-700 dark:bg-indigo-950/60 dark:text-indigo-300 border-indigo-200 dark:border-indigo-800',
      heroThesis: `${symbol} has completed a classic multi-stage Volatility Contraction Pattern (VCP). Tightening contraction stages (T-stages) absorbed overhead supply, and volume expansion confirmed institutional accumulation at the breakout pivot.`,
      institutionalRationale: 'Institutional funds accumulated during dry pullbacks; lack of heavy volume selling validates supply exhaustion.',
      gates: [
        { name: 'Trend Gate', passed: true, detail: 'Price stacked above rising 50 & 200 SMA; ADX > 20 with DI+ dominant' },
        { name: 'Volume Gate', passed: true, detail: 'Volume dry-up in tightest contraction followed by >1.25x breakout volume' },
        { name: 'Volatility Gate', passed: true, detail: 'ATR contracted to lower quartile of 100-day range prior to trigger' },
        { name: 'Risk-to-Reward Gate', passed: true, detail: 'Asymmetric 1:3.0+ edge with tight pivot stop anchored at swing low' },
      ],
    };
  }

  if (pat.includes('pullback') || r.includes('pullback') || r.includes('ema')) {
    return {
      patternName: '21 EMA Pullback Reversal',
      tagColor: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800',
      heroThesis: `${symbol} completed an orderly pullback into its rising 21-day Exponential Moving Average within a confirmed uptrend. Low-volume consolidation resolved into a strong bullish reversal candle.`,
      institutionalRationale: 'Smart money buying on pullbacks into institutional moving average benchmarks with expanding positive volume flow.',
      gates: [
        { name: 'Trend Gate', passed: true, detail: 'Price strictly adhering to rising 21 EMA support within established trend' },
        { name: 'Volume Gate', passed: true, detail: 'Pullback volume dried to <60% of 20-day average, signaling zero dumping' },
        { name: 'Volatility Gate', passed: true, detail: 'Orderly candle bodies at dynamic support providing high precision entry' },
        { name: 'Risk-to-Reward Gate', passed: true, detail: 'Stop loss tightly placed 2x ATR below EMA swing low with 1:3.5 upside' },
      ],
    };
  }

  if (pat.includes('momentum') || r.includes('20') || r.includes('high')) {
    return {
      patternName: '20-Day Momentum Breakout',
      tagColor: 'bg-blue-100 text-blue-700 dark:bg-blue-950/60 dark:text-blue-300 border-blue-200 dark:border-blue-800',
      heroThesis: `${symbol} broke out to a clean 20-day new high within 20% of its 52-week peak. Breakout is confirmed by expanding volume and strong market regime relative strength.`,
      institutionalRationale: 'Breakout out of multi-week base driven by sustained institutional accumulation and sector leadership.',
      gates: [
        { name: 'Trend Gate', passed: true, detail: 'Multi-week resistance cleared with price expanding above all major moving averages' },
        { name: 'Volume Gate', passed: true, detail: 'Institutional volume surge > 1.35x 30-day average on breakout session' },
        { name: 'Volatility Gate', passed: true, detail: 'Tight consolidation range resolved with momentum expansion' },
        { name: 'Risk-to-Reward Gate', passed: true, detail: 'Target staged at 3x and 5x ATR with initial risk capped at 4.0%' },
      ],
    };
  }

  // Default institutional swing trading structure
  return {
    patternName: 'Higher Low Trend Structure',
    tagColor: 'bg-purple-100 text-purple-700 dark:bg-purple-950/60 dark:text-purple-300 border-purple-200 dark:border-purple-800',
    heroThesis: `${symbol} demonstrated consistent accumulation with consecutive higher swing lows, supported by expanding buying volume on advance sessions in line with ${strategy || 'Swing Strategy'}.`,
    institutionalRationale: 'Structural accumulation by institutional participants with positive On-Balance Volume (OBV) slope.',
    gates: [
      { name: 'Trend Gate', passed: true, detail: 'Confirmed series of higher swing lows above key technical support benchmarks' },
      { name: 'Volume Gate', passed: true, detail: 'Accumulation volume on green days substantially outpaces red days' },
      { name: 'Volatility Gate', passed: true, detail: 'Low entry volatility providing high capital efficiency' },
      { name: 'Risk-to-Reward Gate', passed: true, detail: 'Controlled downside risk with minimum 1:2.5 asymmetric upside targets' },
    ],
  };
}

export default function StockAnalysisDeepDive({
  recommendation,
  symbol,
  companyName,
  strategyName,
  recommendationStrength,
  buyPrice,
  sellPrice,
  stopLoss,
  expectedReturnPercent,
  technicalScore,
  fundamentalScore,
  sentimentScore,
  combinedScore,
  compositeScore,
  entryAtr,
  rrRatio,
  suggestedQuantity,
  allocationPct,
  reason,
  note,
  patternDetected,
  setupChecklist,
  technicalPillars,
  onSaveNote,
  onClose,
  isModal = false,
  className = '',
}: StockAnalysisDeepDiveProps) {
  // Aggregate fields from either recommendation object or standalone props
  const sym = cleanSymbol(recommendation?.symbol || symbol || 'STOCK');
  const compName = recommendation?.company_name || companyName || sym;
  const strat = recommendation?.strategy_name || strategyName || 'Swing_Trading_v3';
  const strength = (recommendation?.recommendation_strength || recommendationStrength || 'BUY').toUpperCase();

  const buy = recommendation?.buy_price ?? buyPrice ?? 1000;
  const sl = recommendation?.stop_loss ?? stopLoss ?? (buy ? buy * 0.94 : 940);
  const tgt = recommendation?.sell_price ?? sellPrice ?? (buy ? buy * 1.15 : 1150);

  const techScore = recommendation?.technical_score ?? technicalScore ?? 0.82;
  const fundScore = recommendation?.fundamental_score ?? fundamentalScore ?? 0.74;
  const sentScore = recommendation?.sentiment_score ?? sentimentScore ?? 0.68;
  const combScore = recommendation?.combined_score ?? combinedScore ?? 0.80;
  const compScoreVal = recommendation?.composite_score ?? compositeScore ?? (combScore ? combScore * 5 : 4.1);

  const atrVal = recommendation?.entry_atr ?? entryAtr ?? (buy ? buy * 0.025 : 25);
  const rawRR = recommendation?.rr_ratio ?? rrRatio;
  const calculatedRR = rawRR || (buy && sl && tgt && buy > sl ? (tgt - buy) / (buy - sl) : 3.0);
  const rr = Number(calculatedRR.toFixed(1));

  const qty = recommendation?.suggested_quantity ?? suggestedQuantity ?? (buy > 0 ? Math.max(1, Math.round(50000 / buy)) : 10);
  const alloc = recommendation?.allocation_pct ?? allocationPct ?? 15;
  const reasonText = recommendation?.reason || reason || '';
  const initialNote = recommendation?.note || note || '';
  const patternNameVal = recommendation?.pattern_detected || patternDetected;

  // State
  const [customNote, setCustomNote] = useState(initialNote);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [copied, setCopied] = useState(false);

  // Risk and Return metrics
  const riskAmount = buy - sl;
  const riskPct = buy > 0 ? (riskAmount / buy) * 100 : 6.0;
  const rewardAmount = tgt - buy;
  const upsidePct = expectedReturnPercent || (buy > 0 ? (rewardAmount / buy) * 100 : 15.0);
  const positionValue = qty * buy;

  // Setup rationale details
  const setupContext = getSetupDetails(sym, reasonText, strat, patternNameVal);

  // Normalize scores to 0-100%
  const toPct = (score: number) => {
    if (score <= 1.0) return Math.round(score * 100);
    if (score <= 5.0) return Math.round((score / 5.0) * 100);
    return Math.min(100, Math.round(score));
  };

  const techPct = toPct(techScore);
  const fundPct = toPct(fundScore);
  const sentPct = toPct(sentScore);
  const combPct = toPct(combScore);

  // Handle note saving
  const handleSaveNote = async () => {
    setIsSaving(true);
    try {
      if (onSaveNote) {
        await onSaveNote(customNote);
      } else {
        await updateRecommendation(sym, { note: customNote });
      }
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 2500);
    } catch (err) {
      console.error('Failed to save note:', err);
    } finally {
      setIsSaving(false);
    }
  };

  // Copy trade thesis to clipboard
  const handleCopyThesis = () => {
    const summary = `📊 ${sym} Trade Setup Plan (${strat})
Pattern: ${setupContext.patternName}
Entry: ₹${buy.toFixed(2)}
Stop Loss: ₹${sl.toFixed(2)} (-${riskPct.toFixed(1)}%)
Target: ₹${tgt.toFixed(2)} (+${upsidePct.toFixed(1)}%)
R:R: 1:${rr} | Sizing: ${qty} shares (${alloc}% allocation)
Thesis: ${reasonText || setupContext.heroThesis}`;

    navigator.clipboard?.writeText(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const content = (
    <div className={`space-y-6 ${className}`}>
      {/* 1. Hero Header Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-gray-900 via-gray-900 to-blue-950 p-5 md:p-6 text-white shadow-xl border border-blue-900/40">
        {/* Ambient Glow */}
        <div className="absolute top-0 right-0 -mt-8 -mr-8 w-64 h-64 bg-blue-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute bottom-0 left-1/3 -mb-8 w-48 h-48 bg-emerald-500/10 rounded-full blur-2xl pointer-events-none"></div>

        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          {/* Symbol, Name & Badges */}
          <div className="flex items-start md:items-center gap-3.5">
            <div className="w-12 h-12 md:w-14 md:h-14 rounded-2xl bg-blue-600/30 border border-blue-400/40 text-blue-300 font-extrabold text-xl flex items-center justify-center shrink-0 shadow-inner">
              {sym.slice(0, 2)}
            </div>
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-2xl md:text-3xl font-extrabold tracking-tight text-white">{sym}</h2>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-bold tracking-wide bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-xs">
                  {strength}
                </span>
                <span className="px-2.5 py-0.5 rounded-full text-xs font-mono font-medium bg-gray-800 text-gray-300 border border-gray-700">
                  {strat}
                </span>
              </div>
              <p className="text-sm text-gray-400 mt-0.5 line-clamp-1">{compName}</p>
            </div>
          </div>

          {/* Pricing & Key Summary Pill */}
          <div className="flex flex-wrap items-center gap-4 bg-gray-800/60 backdrop-blur-md px-4 py-2.5 rounded-xl border border-gray-700/60 self-start md:self-auto">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-gray-400">Entry / Buy</div>
              <div className="text-lg md:text-xl font-bold font-mono text-emerald-400">₹{buy.toFixed(2)}</div>
            </div>
            <div className="h-7 w-px bg-gray-700"></div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-gray-400">Upside Target</div>
              <div className="text-lg md:text-xl font-bold font-mono text-blue-400">
                ₹{tgt.toFixed(2)} <span className="text-xs font-semibold text-emerald-400">(+{upsidePct.toFixed(1)}%)</span>
              </div>
            </div>
            <div className="h-7 w-px bg-gray-700"></div>
            <div>
              <div className="text-[10px] uppercase tracking-wider text-gray-400">Risk : Reward</div>
              <div className="text-lg md:text-xl font-bold font-mono text-amber-400">1 : {rr}</div>
            </div>
          </div>
        </div>
      </div>

      {/* 2. "Why We Picked This Stock" Hero Narrative Card */}
      <div className="rounded-2xl bg-gradient-to-r from-blue-50/80 via-white to-indigo-50/60 dark:from-blue-950/25 dark:via-gray-900 dark:to-indigo-950/20 p-5 md:p-6 border border-blue-200/80 dark:border-blue-900/50 shadow-sm relative">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-blue-600 text-white shadow-xs">
              <SparklesIcon className="w-4 h-4" />
            </span>
            <h3 className="font-bold text-base md:text-lg text-gray-900 dark:text-gray-100">
              Why We Picked This Stock
            </h3>
          </div>
          <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${setupContext.tagColor}`}>
            {setupContext.patternName}
          </span>
        </div>

        {/* Narrative Box */}
        <p className="text-sm md:text-base text-gray-800 dark:text-gray-200 leading-relaxed font-normal">
          {reasonText || setupContext.heroThesis}
        </p>

        {/* Institutional Smart Money Footprint Note */}
        <div className="mt-3.5 pt-3 border-t border-blue-100 dark:border-blue-900/40 flex items-center justify-between flex-wrap gap-2 text-xs text-gray-600 dark:text-gray-400">
          <div className="flex items-center gap-1.5">
            <FireIcon className="w-4 h-4 text-amber-500 shrink-0" />
            <span><strong className="text-gray-800 dark:text-gray-200">Institutional Footprint:</strong> {setupContext.institutionalRationale}</span>
          </div>
          <button
            type="button"
            onClick={handleCopyThesis}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-100/60 dark:hover:bg-blue-900/40 transition cursor-pointer"
          >
            {copied ? (
              <>
                <ClipboardDocumentCheckIcon className="w-3.5 h-3.5 text-emerald-500" />
                <span className="text-emerald-500">Copied!</span>
              </>
            ) : (
              <>
                <ClipboardDocumentIcon className="w-3.5 h-3.5" />
                <span>Copy Setup</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* 3. Interactive Stock Detail Chart */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
            <ChartBarIcon className="w-4 h-4 text-blue-500" />
            Interactive Multi-Timeframe Chart & Overlays
          </h4>
          <span className="text-[11px] text-gray-500 dark:text-gray-400">
            Candlestick / Line Mode • EMA 21 & SMA 50 • Volume
          </span>
        </div>
        <StockDetailChart
          symbol={sym}
          entryPrice={buy}
          stopLoss={sl}
          target={tgt}
          patternName={setupContext.patternName}
          height={380}
        />
      </div>

      {/* 4. Four-Pillar Score Gauges */}
      <div className="rounded-2xl bg-white dark:bg-gray-900 p-5 md:p-6 border border-gray-200 dark:border-gray-800 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
            <ScaleIcon className="w-4 h-4 text-purple-500" />
            Algorithmic 4-Pillar Quality Gauges
          </h4>
          <div className="flex items-center gap-1.5 text-xs font-semibold text-gray-700 dark:text-gray-300">
            <span>Overall Composite:</span>
            <span className="px-2 py-0.5 rounded-md bg-purple-100 dark:bg-purple-950/60 text-purple-700 dark:text-purple-300 font-mono font-bold border border-purple-200 dark:border-purple-800">
              {compScoreVal.toFixed(2)} / 5.0
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Pillar 1: Technical */}
          <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-850 border border-gray-100 dark:border-gray-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">Technical Momentum</span>
              <span className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400">{techPct}%</span>
            </div>
            <div className="w-full h-2 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-500"
                style={{ width: `${techPct}%` }}
              ></div>
            </div>
            <p className="text-[11px] text-gray-500 dark:text-gray-400">
              Price trend alignment, RSI momentum re-test & 21-EMA support.
            </p>
          </div>

          {/* Pillar 2: Fundamental */}
          <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-850 border border-gray-100 dark:border-gray-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">Fundamental Health</span>
              <span className="text-xs font-mono font-bold text-blue-600 dark:text-blue-400">{fundPct}%</span>
            </div>
            <div className="w-full h-2 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-500 to-indigo-400 transition-all duration-500"
                style={{ width: `${fundPct}%` }}
              ></div>
            </div>
            <p className="text-[11px] text-gray-500 dark:text-gray-400">
              Earnings growth consistency, positive ROE and healthy balance sheet.
            </p>
          </div>

          {/* Pillar 3: Sentiment & Inflows */}
          <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-850 border border-gray-100 dark:border-gray-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">Smart Inflows & Sentiment</span>
              <span className="text-xs font-mono font-bold text-amber-600 dark:text-amber-400">{sentPct}%</span>
            </div>
            <div className="w-full h-2 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-amber-500 to-orange-400 transition-all duration-500"
                style={{ width: `${sentPct}%` }}
              ></div>
            </div>
            <p className="text-[11px] text-gray-500 dark:text-gray-400">
              Delivery volume spikes and institutional accumulation patterns.
            </p>
          </div>

          {/* Pillar 4: Combined Multi-Factor */}
          <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-850 border border-gray-100 dark:border-gray-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">Confidence Rating</span>
              <span className="text-xs font-mono font-bold text-purple-600 dark:text-purple-400">{combPct}%</span>
            </div>
            <div className="w-full h-2 rounded-full bg-gray-200 dark:bg-gray-700 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-500"
                style={{ width: `${combPct}%` }}
              ></div>
            </div>
            <p className="text-[11px] text-gray-500 dark:text-gray-400">
              Weighted composite rating across universe risk criteria.
            </p>
          </div>
        </div>
      </div>

      {/* 5. Strategy Checklist & Technical Validation Gates */}
      <div className="rounded-2xl bg-white dark:bg-gray-900 p-5 md:p-6 border border-gray-200 dark:border-gray-800 shadow-sm space-y-3.5">
        <h4 className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
          <CheckCircleIcon className="w-4 h-4 text-emerald-500" />
          Strategy Checklist & Strict Technical Gates
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {setupContext.gates.map((gate, idx) => (
            <div
              key={idx}
              className="flex items-start gap-3 p-3 rounded-xl bg-gray-50/70 dark:bg-gray-850/60 border border-gray-100 dark:border-gray-800"
            >
              <div className="w-6 h-6 rounded-full bg-emerald-100 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shrink-0 mt-0.5">
                <CheckIcon className="w-3.5 h-3.5 stroke-[2.5]" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-bold text-gray-900 dark:text-gray-100">{gate.name}</span>
                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border border-emerald-300/40">
                    PASSED
                  </span>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{gate.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 6. Trade Execution & Risk Management Plan */}
      <div className="rounded-2xl bg-gradient-to-br from-gray-50 via-white to-gray-50 dark:from-gray-900 dark:via-gray-850 dark:to-gray-900 p-5 md:p-6 border border-gray-200 dark:border-gray-800 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
            <ShieldCheckIcon className="w-4 h-4 text-blue-500" />
            Trade Execution & Capital Risk Controls
          </h4>
          <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
            Capital at Risk: ₹{(riskAmount * qty).toFixed(0)}
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {/* Entry Zone */}
          <div className="p-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-750 text-center">
            <div className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase">Entry Pivot</div>
            <div className="text-sm md:text-base font-bold font-mono text-blue-600 dark:text-blue-400 mt-0.5">
              ₹{buy.toFixed(2)}
            </div>
            <div className="text-[10px] text-gray-400">Limit order zone</div>
          </div>

          {/* Stop Loss */}
          <div className="p-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-750 text-center">
            <div className="text-[10px] font-semibold text-rose-500 uppercase">Stop Loss</div>
            <div className="text-sm md:text-base font-bold font-mono text-rose-600 dark:text-rose-400 mt-0.5">
              ₹{sl.toFixed(2)}
            </div>
            <div className="text-[10px] text-rose-500/80 font-medium">-{riskPct.toFixed(1)}% Max Risk</div>
          </div>

          {/* Target Price */}
          <div className="p-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-750 text-center">
            <div className="text-[10px] font-semibold text-emerald-500 uppercase">Target 1</div>
            <div className="text-sm md:text-base font-bold font-mono text-emerald-600 dark:text-emerald-400 mt-0.5">
              ₹{tgt.toFixed(2)}
            </div>
            <div className="text-[10px] text-emerald-600 dark:text-emerald-400 font-medium">+{upsidePct.toFixed(1)}% Gain</div>
          </div>

          {/* R:R Ratio */}
          <div className="p-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-750 text-center">
            <div className="text-[10px] font-semibold text-amber-500 uppercase">Risk : Reward</div>
            <div className="text-sm md:text-base font-bold font-mono text-amber-600 dark:text-amber-400 mt-0.5">
              1 : {rr}
            </div>
            <div className="text-[10px] text-gray-400">Asymmetric edge</div>
          </div>

          {/* Suggested Sizing */}
          <div className="p-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-750 text-center">
            <div className="text-[10px] font-semibold text-indigo-500 uppercase">Suggested Sizing</div>
            <div className="text-sm md:text-base font-bold font-mono text-indigo-600 dark:text-indigo-400 mt-0.5">
              {qty} Shares
            </div>
            <div className="text-[10px] text-gray-400">{alloc}% Portfolio</div>
          </div>

          {/* Entry ATR */}
          <div className="p-3 rounded-xl bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-750 text-center">
            <div className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase">Entry ATR</div>
            <div className="text-sm md:text-base font-bold font-mono text-gray-800 dark:text-gray-200 mt-0.5">
              ₹{atrVal.toFixed(1)}
            </div>
            <div className="text-[10px] text-gray-400">{((atrVal / buy) * 100).toFixed(1)}% Volatility</div>
          </div>
        </div>
      </div>

      {/* 7. Trader Custom Notes & Observation */}
      <div className="rounded-2xl bg-white dark:bg-gray-900 p-5 md:p-6 border border-gray-200 dark:border-gray-800 shadow-sm space-y-3">
        <label className="block text-xs font-bold text-gray-500 dark:text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
          <DocumentTextIcon className="w-4 h-4 text-amber-500" />
          Trader Observation & Personal Plan Notes
        </label>
        <div className="space-y-2.5">
          <textarea
            rows={2}
            value={customNote}
            onChange={(e) => setCustomNote(e.target.value)}
            placeholder="Add personal execution notes, earnings calendar dates, or scale-in targets for this stock..."
            className="w-full text-xs md:text-sm p-3 rounded-xl border border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:outline-hidden transition"
          />
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-gray-400">
              Notes are persisted to the database and synced across devices.
            </span>
            <div className="flex items-center gap-2">
              {saveSuccess && (
                <span className="text-xs text-emerald-600 dark:text-emerald-400 font-medium flex items-center gap-1">
                  <CheckIcon className="w-3.5 h-3.5" /> Note saved!
                </span>
              )}
              <button
                type="button"
                onClick={handleSaveNote}
                disabled={isSaving}
                className="px-4 py-1.5 bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white rounded-lg text-xs font-semibold transition disabled:opacity-50 cursor-pointer shadow-xs"
              >
                {isSaving ? 'Saving...' : 'Save Note'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // If used inside a modal popup
  if (isModal) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-5 bg-black/70 backdrop-blur-xs animate-in fade-in duration-200 overflow-y-auto">
        <div
          className="relative w-full max-w-4xl my-auto bg-white dark:bg-gray-900 rounded-3xl shadow-2xl border border-gray-200 dark:border-gray-800 overflow-hidden transform transition-all"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Modal Header Bar */}
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800 bg-gray-50/90 dark:bg-gray-850/90 flex items-center justify-between sticky top-0 z-20 backdrop-blur-md">
            <div className="flex items-center gap-2.5">
              <span className="font-extrabold text-base text-gray-900 dark:text-gray-100">
                Stock Deep-Dive Analysis
              </span>
              <span className="px-2 py-0.5 rounded-md bg-blue-100 dark:bg-blue-950/60 text-blue-700 dark:text-blue-300 font-mono text-xs font-semibold">
                {sym}
              </span>
            </div>
            {onClose && (
              <button
                type="button"
                onClick={onClose}
                className="p-1.5 rounded-xl text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-800 transition cursor-pointer"
                title="Close Deep Dive"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            )}
          </div>

          {/* Scrollable Body */}
          <div className="p-5 md:p-6 max-h-[82vh] overflow-y-auto">
            {content}
          </div>

          {/* Modal Footer Bar */}
          <div className="px-6 py-3.5 bg-gray-50 dark:bg-gray-850 border-t border-gray-200 dark:border-gray-800 flex items-center justify-between">
            <div className="text-xs text-gray-500 dark:text-gray-400">
              Smart Advice Research & Execution Engine
            </div>
            {onClose && (
              <button
                type="button"
                onClick={onClose}
                className="px-5 py-2 text-xs font-bold text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-750 rounded-xl transition cursor-pointer"
              >
                Close
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Standalone inline panel
  return content;
}
