'use client';

import React from 'react';
import {
  ShieldCheckIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon,
  ChartBarIcon,
  ScaleIcon,
  AdjustmentsHorizontalIcon,
  BanknotesIcon,
  FireIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline';

export interface UltimatePhasesData {
  confidence_score?: {
    total_score?: number;
    confidence_level?: string;
    action?: string;
    edge_verified?: boolean;
    base_cagr?: number;
    realistic_cagr?: number;
    haircut?: number;
    component_scores?: Record<string, number>;
    weights?: Record<string, number>;
  };
  validation?: {
    edge_verified?: boolean;
    tests_passed?: number;
    total_tests?: number;
    overall_confidence_pct?: number;
    dsr?: {
      sr_observed?: number;
      sr_threshold?: number;
      dsr?: number;
      p_value?: number;
      significant?: boolean;
      confidence_pct?: number;
    };
    monte_carlo_permutation?: {
      actual_sharpe?: number;
      p_value?: number;
      z_score?: number;
      significant?: boolean;
    };
    minimum_track_record?: {
      actual_days?: number;
      actual_years?: number;
      required_days?: number;
      required_years?: number;
      sufficient?: boolean;
    };
  };
  stress_tests?: {
    regime_tests?: Array<{
      regime: string;
      cagr: number;
      max_dd: number;
      sharpe?: number;
      win_rate?: number;
      total_trades?: number;
      min_cagr_required?: number;
      max_dd_tolerance?: number;
      passed: boolean;
    }>;
    cost_sensitivity?: Array<{
      scenario: string;
      brokerage: number;
      slippage: number;
      cagr: number;
      impact_vs_base: number;
      total_trades?: number;
    }>;
    param_sensitivity?: Array<{
      param: string;
      value: any;
      cagr: number;
      total_trades?: number;
      passed?: boolean;
    }>;
  };
  trade_diagnostics?: Record<string, any>;
  walk_forward?: Record<string, any> | null;
}

interface UltimatePhasesViewProps {
  phases: UltimatePhasesData;
  strategyName?: string;
}

export default function UltimatePhasesView({ phases, strategyName }: UltimatePhasesViewProps) {
  const conf = phases.confidence_score || {};
  const val = phases.validation || {};
  const stress = phases.stress_tests || {};

  const totalScore = conf.total_score ?? 0;
  const confidenceLevel = conf.confidence_level || 'Low';
  const isEdgeVerified = conf.edge_verified ?? val.edge_verified ?? false;

  // Level color tokens
  const isHigh = confidenceLevel.toLowerCase() === 'high';
  const isMedium = confidenceLevel.toLowerCase() === 'medium';
  const levelBadgeColor = isHigh
    ? 'bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/30'
    : isMedium
    ? 'bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/30'
    : 'bg-rose-500/15 text-rose-600 dark:text-rose-400 border-rose-500/30';

  const componentLabels: Record<string, string> = {
    walk_forward: 'Walk-Forward Stability',
    dsr: 'Deflated Sharpe Ratio (DSR)',
    mc_permutation: 'Monte Carlo Permutation',
    stress_tests: 'Market Regime Stress Tests',
    param_stability: 'Parameter Stability & Plateau',
    cost_resilience: 'Transaction Cost Resilience',
    data_sufficiency: 'Sample Data Sufficiency',
  };

  const compScores = conf.component_scores || {};
  const compWeights = conf.weights || {};

  return (
    <div className="space-y-6">
      {/* Top Banner: Composite Confidence Score & Robustness Gauge */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-gray-900 via-indigo-950/80 to-purple-950 p-6 sm:p-8 text-white shadow-xl border border-indigo-500/20">
        <div className="absolute top-0 right-0 -mr-16 -mt-16 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 -ml-16 -mb-16 w-64 h-64 bg-purple-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider bg-white/10 backdrop-blur-md border border-white/10 text-indigo-200">
              <ShieldCheckIcon className="w-4 h-4 text-indigo-300" />
              6-Phase Statistical Edge Verification
            </div>
            <h2 className="text-2xl sm:text-3xl font-black tracking-tight">
              Strategy Confidence & Robustness Matrix
            </h2>
            <p className="text-sm text-gray-300 max-w-2xl leading-relaxed">
              Synthesizes walk-forward validation, deflated Sharpe ratios, regime-specific stress tests,
              and transaction cost slippage to verify true alpha versus statistical overfitting.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg border text-xs font-bold ${levelBadgeColor}`}>
                {confidenceLevel} Confidence ({totalScore.toFixed(1)}/100)
              </span>

              <span
                className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-lg border text-xs font-semibold ${
                  isEdgeVerified
                    ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                    : 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                }`}
              >
                {isEdgeVerified ? <CheckCircleIcon className="w-4 h-4" /> : <ExclamationTriangleIcon className="w-4 h-4" />}
                {isEdgeVerified ? 'Statistical Edge Confirmed' : 'Edge Not Statistically Verified'}
              </span>

              {conf.action && (
                <span className="text-xs px-3 py-1 rounded-lg bg-white/10 text-gray-200 border border-white/10">
                  Recommendation: <strong className="text-white">{conf.action}</strong>
                </span>
              )}
            </div>
          </div>

          {/* Radial Score Showcase */}
          <div className="flex items-center gap-6 bg-black/40 backdrop-blur-md p-5 rounded-2xl border border-white/10 self-start lg:self-center shrink-0">
            <div className="text-center">
              <div className="text-4xl sm:text-5xl font-black tracking-tight bg-gradient-to-r from-white via-indigo-200 to-indigo-400 bg-clip-text text-transparent">
                {totalScore.toFixed(0)}
                <span className="text-xl text-gray-400 font-normal">/100</span>
              </div>
              <span className="text-xs uppercase tracking-wider font-semibold text-gray-400 mt-1 block">
                Composite Score
              </span>
            </div>

            <div className="w-px h-12 bg-white/15" />

            <div className="space-y-1 text-xs">
              <div className="text-gray-400">Base CAGR: <strong className="text-white">{(conf.base_cagr ?? 0).toFixed(1)}%</strong></div>
              <div className="text-gray-400">Haircut: <strong className="text-amber-300">{((conf.haircut ?? 0) * 100).toFixed(0)}%</strong></div>
              <div className="text-gray-400">Realistic Projection: <strong className="text-emerald-300">{(conf.realistic_cagr ?? 0).toFixed(1)}%</strong></div>
            </div>
          </div>
        </div>

        {/* 7 Component Progress Bars */}
        {Object.keys(compScores).length > 0 && (
          <div className="mt-8 pt-6 border-t border-white/10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(compScores).map(([key, score]) => {
              const weight = compWeights[key] ?? 0;
              const label = componentLabels[key] || key.replace(/_/g, ' ');
              const pct = Math.min(Math.max(score, 0), 100);

              let barColor = 'bg-rose-500';
              if (pct >= 70) barColor = 'bg-emerald-500';
              else if (pct >= 40) barColor = 'bg-amber-500';

              return (
                <div key={key} className="bg-white/5 rounded-xl p-3 border border-white/5 space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-gray-200 truncate pr-2" title={label}>
                      {label}
                    </span>
                    <span className="font-bold text-white shrink-0">
                      {score.toFixed(1)} <span className="text-gray-400 text-[10px]">({(weight * 100).toFixed(0)}%)</span>
                    </span>
                  </div>
                  <div className="w-full bg-white/10 rounded-full h-1.5 overflow-hidden">
                    <div className={`h-full rounded-full ${barColor}`} style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Grid: Statistical Validation & Minimum Track Record */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Deflated Sharpe Ratio Card */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 space-y-4">
          <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-purple-50 dark:bg-purple-950/60 text-purple-600 dark:text-purple-400">
                <ScaleIcon className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-gray-900 dark:text-white text-base">
                  Deflated Sharpe Ratio (DSR)
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Corrects for selection bias across multiple trial evaluations (Bailey & López de Prado)
                </p>
              </div>
            </div>
            <span
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold ${
                val.dsr?.significant
                  ? 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800'
                  : 'bg-rose-50 dark:bg-rose-950/50 text-rose-600 dark:text-rose-400 border border-rose-200 dark:border-rose-800'
              }`}
            >
              {val.dsr?.significant ? 'Passed (Significant)' : 'Failed (Not Significant)'}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
              <span className="text-[11px] text-gray-500 dark:text-gray-400 block">Observed Sharpe</span>
              <span className="text-lg font-bold text-gray-900 dark:text-white">
                {Number(val.dsr?.sr_observed ?? 0).toFixed(3)}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
              <span className="text-[11px] text-gray-500 dark:text-gray-400 block">DSR Threshold</span>
              <span className="text-lg font-bold text-gray-900 dark:text-white">
                {Number(val.dsr?.sr_threshold ?? 0).toFixed(3)}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
              <span className="text-[11px] text-gray-500 dark:text-gray-400 block">p-Value</span>
              <span className="text-lg font-bold text-gray-900 dark:text-white">
                {Number(val.dsr?.p_value ?? 1.0).toFixed(3)}
              </span>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-indigo-50/50 dark:bg-indigo-950/30 border border-indigo-100 dark:border-indigo-800/40 text-xs text-indigo-900 dark:text-indigo-200 leading-relaxed">
            DSR tests whether the observed Sharpe ratio remains statistically superior to zero after penalizing for non-normal returns, skewness, and the number of trials tested.
          </div>
        </div>

        {/* Monte Carlo Permutation Card */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 space-y-4">
          <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-blue-50 dark:bg-blue-950/60 text-blue-600 dark:text-blue-400">
                <CpuChipIcon className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-gray-900 dark:text-white text-base">
                  Monte Carlo Permutation Test
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Reshuffles return sequences to test whether timing skill was luck
                </p>
              </div>
            </div>
            <span
              className={`px-2.5 py-1 rounded-lg text-xs font-semibold ${
                val.monte_carlo_permutation?.significant
                  ? 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-600 dark:text-emerald-400 border border-emerald-200 dark:border-emerald-800'
                  : 'bg-amber-50 dark:bg-amber-950/50 text-amber-600 dark:text-amber-400 border border-amber-200 dark:border-amber-800'
              }`}
            >
              {val.monte_carlo_permutation?.significant ? 'Passed' : 'p > 0.05 (Unverified)'}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
              <span className="text-[11px] text-gray-500 dark:text-gray-400 block">Permutation p-Value</span>
              <span className="text-lg font-bold text-gray-900 dark:text-white">
                {Number(val.monte_carlo_permutation?.p_value ?? 1.0).toFixed(3)}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
              <span className="text-[11px] text-gray-500 dark:text-gray-400 block">Z-Score</span>
              <span className="text-lg font-bold text-gray-900 dark:text-white">
                {Number(val.monte_carlo_permutation?.z_score ?? 0).toFixed(2)}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-gray-50 dark:bg-gray-900/60 border border-gray-100 dark:border-gray-700">
              <span className="text-[11px] text-gray-500 dark:text-gray-400 block">Actual Track Record</span>
              <span className="text-lg font-bold text-gray-900 dark:text-white">
                {val.minimum_track_record?.actual_years ? `${val.minimum_track_record.actual_years} Years` : '1.0 Year'}
              </span>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
            Tests 1,000 synthetic return permutations with auto-correlation adjustment to eliminate spurious regression artifacts.
          </div>
        </div>
      </div>

      {/* Regime-Specific Stress Testing Matrix */}
      {stress.regime_tests && stress.regime_tests.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-gray-100 dark:border-gray-700 pb-4">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-xl bg-amber-50 dark:bg-amber-950/60 text-amber-600 dark:text-amber-400">
                <FireIcon className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-gray-900 dark:text-white text-base">
                  Historical Regime Stress Tests
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Evaluates strategy resilience across severe market shocks and bull/bear supercycles
                </p>
              </div>
            </div>
            <span className="text-xs font-semibold px-3 py-1 rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300">
              {stress.regime_tests.filter((r) => r.passed).length} / {stress.regime_tests.length} Regimes Passed
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {stress.regime_tests.map((r, idx) => {
              const isPositive = r.cagr >= 0;
              return (
                <div
                  key={idx}
                  className={`p-4 rounded-xl border transition-all ${
                    r.passed
                      ? 'bg-emerald-50/30 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-800/60'
                      : 'bg-rose-50/30 dark:bg-rose-950/20 border-rose-200 dark:border-rose-800/60'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm text-gray-900 dark:text-white">{r.regime}</span>
                    <span
                      className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded ${
                        r.passed
                          ? 'bg-emerald-500/20 text-emerald-600 dark:text-emerald-400'
                          : 'bg-rose-500/20 text-rose-600 dark:text-rose-400'
                      }`}
                    >
                      {r.passed ? 'Pass' : 'Fail'}
                    </span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-gray-100 dark:border-gray-700/60">
                    <div>
                      <span className="text-[10px] text-gray-500 dark:text-gray-400 block">Regime CAGR</span>
                      <span className={`text-base font-bold ${isPositive ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                        {isPositive ? '+' : ''}{r.cagr.toFixed(1)}%
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-gray-500 dark:text-gray-400 block">Max Drawdown</span>
                      <span className="text-base font-bold text-amber-600 dark:text-amber-400">
                        {r.max_dd.toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-gray-500 dark:text-gray-400 mt-2">
                    <span>Target: &gt;={r.min_cagr_required ?? 0}%</span>
                    <span>Tolerance: &lt;={r.max_dd_tolerance ?? 0}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Transaction Cost Resilience Table */}
      {stress.cost_sensitivity && stress.cost_sensitivity.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 space-y-4">
          <div className="flex items-center gap-2.5 border-b border-gray-100 dark:border-gray-700 pb-4">
            <div className="p-2 rounded-xl bg-emerald-50 dark:bg-emerald-950/60 text-emerald-600 dark:text-emerald-400">
              <BanknotesIcon className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-gray-900 dark:text-white text-base">
                Transaction Cost & Slippage Sensitivity
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Tests edge survival under increasing broker commissions, exchange turnover, and execution slippage
              </p>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-gray-50 dark:bg-gray-900 text-gray-600 dark:text-gray-300 uppercase text-xs tracking-wider border-b border-gray-200 dark:border-gray-700 font-semibold">
                <tr>
                  <th className="py-3 px-4">Cost Scenario</th>
                  <th className="py-3 px-4 text-right">Brokerage</th>
                  <th className="py-3 px-4 text-right">Slippage</th>
                  <th className="py-3 px-4 text-right">Net CAGR</th>
                  <th className="py-3 px-4 text-right">Alpha Drag</th>
                  <th className="py-3 px-4 text-right">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-700/80">
                {stress.cost_sensitivity.map((c, idx) => {
                  const isPositive = c.cagr >= 0;
                  return (
                    <tr key={idx} className="hover:bg-gray-50/50 dark:hover:bg-gray-700/30 transition-colors">
                      <td className="py-3 px-4 font-semibold text-gray-900 dark:text-white">
                        {c.scenario}
                      </td>
                      <td className="py-3 px-4 text-right text-gray-600 dark:text-gray-300">
                        {(c.brokerage * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-4 text-right text-gray-600 dark:text-gray-300">
                        {(c.slippage * 100).toFixed(2)}%
                      </td>
                      <td className={`py-3 px-4 text-right font-bold ${isPositive ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                        {isPositive ? '+' : ''}{c.cagr.toFixed(1)}%
                      </td>
                      <td className="py-3 px-4 text-right text-gray-500 dark:text-gray-400 font-medium">
                        {c.impact_vs_base.toFixed(2)} pp
                      </td>
                      <td className="py-3 px-4 text-right">
                        <span
                          className={`px-2 py-0.5 rounded text-xs font-semibold ${
                            isPositive
                              ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300'
                              : 'bg-rose-100 dark:bg-rose-950/60 text-rose-700 dark:text-rose-300'
                          }`}
                        >
                          {isPositive ? 'Profitable' : 'Friction Negative'}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Parameter Stability & Cliff Detection */}
      {stress.param_sensitivity && stress.param_sensitivity.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-gray-200 dark:border-gray-700 space-y-4">
          <div className="flex items-center gap-2.5 border-b border-gray-100 dark:border-gray-700 pb-4">
            <div className="p-2 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400">
              <AdjustmentsHorizontalIcon className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-gray-900 dark:text-white text-base">
                Parameter Sensitivity & Cliff Detection
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Sweeps stop loss multipliers and indicator thresholds to detect curve-fitting cliffs
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {stress.param_sensitivity.map((p, idx) => {
              const isBase = p.param === 'BASE';
              const isPos = p.cagr >= 0;
              return (
                <div
                  key={idx}
                  className={`p-3 rounded-xl border text-xs ${
                    isBase
                      ? 'bg-indigo-50/60 dark:bg-indigo-950/40 border-indigo-200 dark:border-indigo-800'
                      : 'bg-gray-50/80 dark:bg-gray-900/60 border-gray-100 dark:border-gray-700'
                  }`}
                >
                  <span className="font-semibold text-gray-700 dark:text-gray-300 block truncate" title={p.param}>
                    {p.param} = {String(p.value)}
                  </span>
                  <div className="flex items-baseline justify-between mt-1">
                    <span className={`text-base font-bold ${isPos ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
                      {isPos ? '+' : ''}{p.cagr.toFixed(1)}%
                    </span>
                    {p.total_trades && (
                      <span className="text-[10px] text-gray-400">{p.total_trades} trades</span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
