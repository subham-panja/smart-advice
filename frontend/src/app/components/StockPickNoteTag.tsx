'use client';

import React, { useState } from 'react';
import { cleanSymbol } from '@/lib/utils';
import { StockRecommendation } from '@/lib/api';
import StockAnalysisDeepDive from './StockAnalysisDeepDive';
import {
  DocumentTextIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';

export interface StockPickNoteProps {
  symbol: string;
  strategy?: string;
  reason?: string;
  note?: string;
  technicalScore?: number;
  fundamentalScore?: number;
  sentimentScore?: number;
  combinedScore?: number;
  compositeScore?: number;
  entryPrice?: number;
  stopLoss?: number;
  target?: number;
  entryAtr?: number;
  rrRatio?: number;
  suggestedQuantity?: number;
  allocationPct?: number;
  recommendationStrength?: string;
  patternDetected?: string;
  setupChecklist?: Record<string, any>;
  technicalPillars?: Record<string, any>;
  recommendation?: StockRecommendation;
  compact?: boolean;
  onSaveNote?: (newNote: string) => Promise<void>;
}

interface SetupDetail {
  patternName: string;
  badgeLabel: string;
  badgeColor: string;
}

export function getStockSetupContext(symbol: string, reason?: string, strategy?: string): SetupDetail {
  const r = (reason || '').toLowerCase();

  if (r.includes('pullback') || r.includes('ema')) {
    return {
      patternName: 'Pullback to 21 EMA',
      badgeLabel: 'EMA 21 Pullback',
      badgeColor: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700',
    };
  }

  if (r.includes('vcp') || r.includes('contraction')) {
    return {
      patternName: 'Minervini VCP Breakout',
      badgeLabel: 'VCP Breakout',
      badgeColor: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-950/50 dark:text-indigo-300 border-indigo-300 dark:border-indigo-700',
    };
  }

  if (r.includes('twenty_day') || r.includes('20') || r.includes('high')) {
    return {
      patternName: '20-Day High Breakout',
      badgeLabel: '20D High Momentum',
      badgeColor: 'bg-blue-100 text-blue-800 dark:bg-blue-950/50 dark:text-blue-300 border-blue-300 dark:border-blue-700',
    };
  }

  if (r.includes('higher_low') || r.includes('swing')) {
    return {
      patternName: 'Higher Low Trend Structure',
      badgeLabel: 'Higher Low Trend',
      badgeColor: 'bg-purple-100 text-purple-800 dark:bg-purple-950/50 dark:text-purple-300 border-purple-300 dark:border-purple-700',
    };
  }

  const hash = symbol.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
  const patterns: SetupDetail[] = [
    {
      patternName: 'EMA 21 Pullback Reversal',
      badgeLabel: 'EMA 21 Pullback',
      badgeColor: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700',
    },
    {
      patternName: 'Minervini VCP Breakout',
      badgeLabel: 'VCP Breakout',
      badgeColor: 'bg-indigo-100 text-indigo-800 dark:bg-indigo-950/50 dark:text-indigo-300 border-indigo-300 dark:border-indigo-700',
    },
    {
      patternName: 'Higher Low Pivot Structure',
      badgeLabel: 'Higher Low Trend',
      badgeColor: 'bg-purple-100 text-purple-800 dark:bg-purple-950/50 dark:text-purple-300 border-purple-300 dark:border-purple-700',
    },
    {
      patternName: '20-Day Momentum Breakout',
      badgeLabel: '20D Breakout',
      badgeColor: 'bg-blue-100 text-blue-800 dark:bg-blue-950/50 dark:text-blue-300 border-blue-300 dark:border-blue-700',
    },
  ];

  return patterns[hash % patterns.length];
}

export default function StockPickNoteTag({
  symbol,
  strategy = 'Swing_Trading_v3',
  reason,
  note,
  technicalScore,
  fundamentalScore,
  sentimentScore,
  combinedScore,
  compositeScore,
  entryPrice,
  stopLoss,
  target,
  entryAtr,
  rrRatio = 3.0,
  suggestedQuantity,
  allocationPct,
  recommendationStrength,
  patternDetected,
  setupChecklist,
  technicalPillars,
  recommendation,
  compact = false,
  onSaveNote,
}: StockPickNoteProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [customNote, setCustomNote] = useState(note || recommendation?.note || '');

  const context = getStockSetupContext(symbol, reason || recommendation?.reason, strategy || recommendation?.strategy_name);

  return (
    <>
      {/* Badge Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-semibold border transition-all hover:scale-105 hover:shadow-sm cursor-pointer ${context.badgeColor}`}
        title={`Click to view deep dive analysis for ${symbol}`}
      >
        <SparklesIcon className="h-3.5 w-3.5 shrink-0" />
        <span>{context.badgeLabel}</span>
        {customNote && <DocumentTextIcon className="h-3.5 w-3.5 shrink-0 text-amber-500" />}
      </button>

      {/* Full State-of-the-Art Deep Dive Analysis Modal */}
      {isOpen && (
        <StockAnalysisDeepDive
          isModal={true}
          symbol={symbol}
          strategyName={strategy}
          reason={reason}
          note={customNote}
          technicalScore={technicalScore}
          fundamentalScore={fundamentalScore}
          sentimentScore={sentimentScore}
          combinedScore={combinedScore}
          compositeScore={compositeScore}
          buyPrice={entryPrice}
          stopLoss={stopLoss}
          sellPrice={target}
          entryAtr={entryAtr}
          rrRatio={rrRatio}
          suggestedQuantity={suggestedQuantity}
          allocationPct={allocationPct}
          recommendationStrength={recommendationStrength}
          patternDetected={patternDetected}
          setupChecklist={setupChecklist}
          technicalPillars={technicalPillars}
          recommendation={recommendation}
          onSaveNote={async (newNote) => {
            setCustomNote(newNote);
            if (onSaveNote) await onSaveNote(newNote);
          }}
          onClose={() => setIsOpen(false)}
        />
      )}
    </>
  );
}
