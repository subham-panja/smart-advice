"use client";

import React, { useEffect, useState, useCallback } from "react";
import {
  getIntradayStatus,
  getIntradayConfig,
  getIntradayPositions,
  getIntradaySignals,
  createIntradayPosition,
  squareOffAllIntraday,
  closeIntradayPosition,
  IntradayClock,
  IntradayDayStats,
  IntradayPosition,
  IntradaySignal,
} from "@/lib/api";
import StockPickNoteTag from "../components/StockPickNoteTag";
import {
  ClockIcon,
  ShieldExclamationIcon,
  FireIcon,
  ArrowTrendingUpIcon,
  ArrowPathIcon,
  AdjustmentsHorizontalIcon,
} from "@heroicons/react/24/outline";

export default function IntradayPage() {
  const [clock, setClock] = useState<IntradayClock | null>(null);
  const [stats, setStats] = useState<IntradayDayStats | null>(null);
  const [config, setConfig] = useState<any>(null);
  const [positions, setPositions] = useState<IntradayPosition[]>([]);
  const [signals, setSignals] = useState<IntradaySignal[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [squareOffLoading, setSquareOffLoading] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [currentTime, setCurrentTime] = useState("");

  // Update IST clock every second
  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setCurrentTime(
        now.toLocaleTimeString("en-IN", {
          timeZone: "Asia/Kolkata",
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        }) + " IST"
      );
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const loadData = useCallback(async () => {
    try {
      setRefreshing(true);
      const [statusRes, configRes, posRes, sigRes] = await Promise.allSettled([
        getIntradayStatus(),
        getIntradayConfig(),
        getIntradayPositions("OPEN"),
        getIntradaySignals(),
      ]);

      if (statusRes.status === "fulfilled" && statusRes.value.status === "success") {
        setClock(statusRes.value.clock);
        setStats(statusRes.value.stats);
      }
      if (configRes.status === "fulfilled" && configRes.value.status === "success") {
        setConfig(configRes.value.config);
      }
      if (posRes.status === "fulfilled" && posRes.value.status === "success") {
        setPositions(posRes.value.positions || []);
      }
      if (sigRes.status === "fulfilled" && sigRes.value.status === "success") {
        setSignals(sigRes.value.signals || []);
      }
    } catch {
      // Fallback cleanly
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 15000);
    return () => clearInterval(interval);
  }, [loadData]);

  const handleSquareOffAll = async () => {
    if (
      !confirm(
        "Are you sure you want to SQUARE OFF ALL active intraday positions immediately?"
      )
    ) {
      return;
    }
    try {
      setSquareOffLoading(true);
      await squareOffAllIntraday("Emergency square-off clicked by user");
      await loadData();
    } catch (err: any) {
      alert("Square off failed: " + (err?.message || err));
    } finally {
      setSquareOffLoading(false);
    }
  };

  const handleClosePosition = async (posId: string) => {
    if (!confirm("Close this intraday position?")) return;
    try {
      await closeIntradayPosition(posId, "Manual close by user");
      await loadData();
    } catch (err: any) {
      alert("Failed to close position: " + (err?.message || err));
    }
  };

  const handleExecuteSignal = async (sig: IntradaySignal) => {
    const qty = prompt(
      `Paper Execute Entry for ${sig.symbol}:\nEntry: ₹${sig.current_price}\nTarget 1: ₹${sig.target_1} | Target 2: ₹${sig.target_2}\nStop Loss: ₹${sig.stop_loss}\n\nEnter Quantity:`,
      "50"
    );
    if (!qty || isNaN(Number(qty)) || Number(qty) <= 0) return;

    try {
      await createIntradayPosition({
        symbol: sig.symbol,
        entry_price: sig.current_price,
        quantity: parseInt(qty, 10),
        stop_loss: sig.stop_loss,
        target_1: sig.target_1,
        target_2: sig.target_2,
        pattern: sig.pattern,
        note: sig.reason,
      });
      await loadData();
    } catch (err: any) {
      alert("Execution failed: " + (err?.message || err));
    }
  };

  const getPhaseBadgeColor = (phase?: string) => {
    switch (phase) {
      case "ACTIVE_TRADING":
        return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
      case "OPENING_RANGE":
        return "bg-amber-500/20 text-amber-300 border-amber-500/30";
      case "SQUARE_OFF_WINDOW":
        return "bg-rose-500/20 text-rose-300 border-rose-500/30 animate-pulse";
      case "MARKET_CLOSED":
        return "bg-zinc-800 text-zinc-400 border-zinc-700";
      default:
        return "bg-blue-500/20 text-blue-400 border-blue-500/30";
    }
  };

  const formatPhaseName = (phase?: string) => {
    if (!phase) return "OFFLINE";
    return phase.replace(/_/g, " ");
  };

  return (
    <div className="min-h-screen bg-[#0d1117] text-zinc-100 p-6 lg:p-8 space-y-6">
      {/* Top Banner / Navigation */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              <FireIcon className="w-6 h-6 text-amber-500" />
              Intraday Multi-Confluence Engine
            </h1>
            <span
              className={`px-3 py-0.5 text-xs font-semibold rounded-full border ${getPhaseBadgeColor(
                clock?.phase
              )}`}
            >
              {formatPhaseName(clock?.phase)}
            </span>
          </div>
          <p className="text-xs text-zinc-400 mt-1">
            Rules-based 15m ORB + CPR + VWAP intraday execution with automated 15:15 IST square-off
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-zinc-900 border border-zinc-800 px-3 py-1.5 rounded-lg text-xs font-mono text-zinc-300">
            <ClockIcon className="w-3.5 h-3.5 text-blue-400" />
            <span>{currentTime || "IST Live"}</span>
          </div>

          <button
            onClick={() => setShowConfigModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 rounded-lg text-xs font-medium border border-zinc-800 transition"
          >
            <AdjustmentsHorizontalIcon className="w-3.5 h-3.5" />
            Config
          </button>

          <button
            onClick={loadData}
            disabled={refreshing}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 rounded-lg text-xs font-medium border border-zinc-800 transition disabled:opacity-50"
          >
            <ArrowPathIcon
              className={`w-3.5 h-3.5 ${refreshing ? "animate-spin text-blue-400" : ""}`}
            />
            Sync
          </button>

          <button
            onClick={handleSquareOffAll}
            disabled={squareOffLoading || positions.length === 0}
            className="flex items-center gap-1.5 px-3.5 py-1.5 bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 rounded-lg text-xs font-semibold border border-rose-500/40 transition disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <ShieldExclamationIcon className="w-4 h-4 text-rose-400" />
            {squareOffLoading ? "Squaring Off..." : "Square Off All"}
          </button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-4 shadow-sm">
          <span className="text-xs font-medium text-zinc-400">Realized Day P&L</span>
          <div className="mt-2 flex items-baseline gap-2">
            <span
              className={`text-2xl font-bold font-mono ${
                (stats?.realized_pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              ₹{(stats?.realized_pnl || 0).toLocaleString("en-IN", { minimumFractionDigits: 2 })}
            </span>
          </div>
          <p className="text-[11px] text-zinc-500 mt-1">Booked from closed trades</p>
        </div>

        <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-4 shadow-sm">
          <span className="text-xs font-medium text-zinc-400">Unrealized Open P&L</span>
          <div className="mt-2 flex items-baseline gap-2">
            <span
              className={`text-2xl font-bold font-mono ${
                (stats?.unrealized_pnl || 0) >= 0 ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              ₹{(stats?.unrealized_pnl || 0).toLocaleString("en-IN", { minimumFractionDigits: 2 })}
            </span>
          </div>
          <p className="text-[11px] text-zinc-500 mt-1">Mark-to-market live positions</p>
        </div>

        <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-4 shadow-sm">
          <span className="text-xs font-medium text-zinc-400">Open Positions</span>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-zinc-100">
              {stats?.open_count ?? positions.length}
            </span>
            <span className="text-xs text-zinc-500">
              / {config?.risk_management?.max_open_positions ?? 3} Max
            </span>
          </div>
          <p className="text-[11px] text-zinc-500 mt-1">
            Max risk per trade: {config?.risk_management?.risk_per_trade_pct ?? 1.0}%
          </p>
        </div>

        <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-4 shadow-sm">
          <span className="text-xs font-medium text-zinc-400">Next Action Deadline</span>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-xl font-bold font-mono text-amber-400">
              {clock?.phase === "ACTIVE_TRADING"
                ? "14:45 IST (No New Entries)"
                : clock?.phase === "SQUARE_OFF_WINDOW"
                ? "15:15 IST (Auto Square-Off)"
                : "09:15 IST (Market Open)"}
            </span>
          </div>
          <p className="text-[11px] text-zinc-500 mt-1">Strict EOD capital preservation</p>
        </div>
      </div>

      {/* Active Positions Table */}
      <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ArrowTrendingUpIcon className="w-4 h-4 text-emerald-400" />
            <h2 className="text-sm font-semibold text-zinc-200">
              Active Intraday Positions ({positions.length})
            </h2>
          </div>
          <span className="text-xs text-zinc-500">15:15 IST Mandatory Square-Off</span>
        </div>

        {positions.length === 0 ? (
          <div className="p-8 text-center text-zinc-500 text-sm">
            No open intraday positions. High confluence setups will appear below for execution.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-zinc-900/90 text-zinc-400 uppercase font-medium border-b border-zinc-800">
                <tr>
                  <th className="py-3 px-4">Symbol</th>
                  <th className="py-3 px-4">Setup Context</th>
                  <th className="py-3 px-4">Qty</th>
                  <th className="py-3 px-4">Entry</th>
                  <th className="py-3 px-4">Current</th>
                  <th className="py-3 px-4">Stop Loss</th>
                  <th className="py-3 px-4">Target 1 (50%)</th>
                  <th className="py-3 px-4">Target 2</th>
                  <th className="py-3 px-4">P&L</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60 text-zinc-300">
                {positions.map((pos) => {
                  const pnl = pos.unrealized_pnl ?? 0;
                  const isProfit = pnl >= 0;
                  return (
                    <tr key={pos._id} className="hover:bg-zinc-800/30 transition">
                      <td className="py-3.5 px-4 font-semibold text-white">
                        {pos.symbol}
                      </td>
                      <td className="py-3.5 px-4">
                        <StockPickNoteTag
                          symbol={pos.symbol}
                          note={
                            pos.note ||
                            pos.reason ||
                            `Intraday 15m ORB + VWAP confluence trigger at ₹${pos.entry_price}`
                          }
                        />
                      </td>
                      <td className="py-3.5 px-4 font-mono">{pos.quantity}</td>
                      <td className="py-3.5 px-4 font-mono">₹{pos.entry_price.toFixed(2)}</td>
                      <td className="py-3.5 px-4 font-mono">
                        ₹{(pos.current_price ?? pos.entry_price).toFixed(2)}
                      </td>
                      <td className="py-3.5 px-4 font-mono text-rose-400">
                        ₹{pos.stop_loss.toFixed(2)}
                      </td>
                      <td className="py-3.5 px-4 font-mono text-emerald-400">
                        ₹{pos.target_1.toFixed(2)}
                      </td>
                      <td className="py-3.5 px-4 font-mono text-emerald-400">
                        ₹{pos.target_2.toFixed(2)}
                      </td>
                      <td
                        className={`py-3.5 px-4 font-mono font-semibold ${
                          isProfit ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {isProfit ? "+" : ""}₹{pnl.toFixed(2)}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <button
                          onClick={() => handleClosePosition(pos._id)}
                          className="px-2.5 py-1 bg-zinc-800 hover:bg-rose-900/40 text-zinc-300 hover:text-rose-300 border border-zinc-700 hover:border-rose-700 rounded text-[11px] transition"
                        >
                          Square Off
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Live Intraday Scanner / Opportunities */}
      <div className="bg-zinc-900/60 border border-zinc-800 rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-zinc-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FireIcon className="w-4 h-4 text-amber-400" />
            <h2 className="text-sm font-semibold text-zinc-200">
              Live Intraday Scanner Feed ({signals.length} Qualified Candidates)
            </h2>
          </div>
          <span className="text-xs text-zinc-500">
            Confluence: ORB 15m + VWAP + CPR Gates
          </span>
        </div>

        {signals.length === 0 ? (
          <div className="p-8 text-center text-zinc-500 text-sm">
            Scanning Nifty 50 / liquid universe... No setup has passed all confluence gates yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-zinc-900/90 text-zinc-400 uppercase font-medium border-b border-zinc-800">
                <tr>
                  <th className="py-3 px-4">Symbol</th>
                  <th className="py-3 px-4">Score</th>
                  <th className="py-3 px-4">Confluence Gates</th>
                  <th className="py-3 px-4">Entry Trigger</th>
                  <th className="py-3 px-4">Target 1 / Target 2</th>
                  <th className="py-3 px-4">Stop Loss (1.2x ATR)</th>
                  <th className="py-3 px-4">Setup Note</th>
                  <th className="py-3 px-4 text-right">Execute</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/60 text-zinc-300">
                {signals.map((sig, idx) => {
                  return (
                    <tr key={idx} className="hover:bg-zinc-800/30 transition">
                      <td className="py-3.5 px-4 font-semibold text-white">
                        {sig.symbol}
                      </td>
                      <td className="py-3.5 px-4">
                        <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          {sig.confluence_score ?? 85}/100
                        </span>
                      </td>
                      <td className="py-3.5 px-4">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] ${
                              sig.current_price > (sig.orb?.orb_high || 0)
                                ? "bg-emerald-500/20 text-emerald-400"
                                : "bg-zinc-800 text-zinc-500"
                            }`}
                          >
                            ORB
                          </span>
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] ${
                              sig.current_price > (sig.current_vwap || 0)
                                ? "bg-emerald-500/20 text-emerald-400"
                                : "bg-zinc-800 text-zinc-500"
                            }`}
                          >
                            VWAP
                          </span>
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] ${
                              sig.current_price > (sig.cpr?.tc || 0)
                                ? "bg-emerald-500/20 text-emerald-400"
                                : "bg-zinc-800 text-zinc-500"
                            }`}
                          >
                            CPR
                          </span>
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] ${
                              (sig.volume_ratio || 0) >= 1.5
                                ? "bg-emerald-500/20 text-emerald-400"
                                : "bg-zinc-800 text-zinc-500"
                            }`}
                          >
                            Vol {(sig.volume_ratio || 1).toFixed(1)}x
                          </span>
                        </div>
                      </td>
                      <td className="py-3.5 px-4 font-mono font-semibold text-white">
                        ₹{sig.current_price.toFixed(2)}
                      </td>
                      <td className="py-3.5 px-4 font-mono text-emerald-400">
                        ₹{sig.target_1.toFixed(2)} / ₹{sig.target_2.toFixed(2)}
                      </td>
                      <td className="py-3.5 px-4 font-mono text-rose-400">
                        ₹{sig.stop_loss.toFixed(2)}
                      </td>
                      <td className="py-3.5 px-4">
                        <StockPickNoteTag
                          symbol={sig.symbol}
                          note={
                            sig.reason ||
                            `15m ORB High breakout with VWAP support & CPR confluence`
                          }
                        />
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <button
                          onClick={() => handleExecuteSignal(sig)}
                          className="px-3 py-1 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded text-xs transition"
                        >
                          Execute
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Config Drawer / Modal */}
      {showConfigModal && config && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="bg-[#161b22] border border-zinc-800 rounded-2xl max-w-2xl w-full p-6 space-y-5 max-h-[85vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-4">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <AdjustmentsHorizontalIcon className="w-5 h-5 text-amber-500" />
                Intraday Strategy Configuration
              </h3>
              <button
                onClick={() => setShowConfigModal(false)}
                className="text-zinc-400 hover:text-white"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4 text-xs text-zinc-300">
              <div>
                <span className="font-semibold text-white">Strategy Name:</span>{" "}
                {config.name} (v{config.version})
              </div>
              <div>
                <span className="font-semibold text-white">Description:</span>{" "}
                {config.description}
              </div>

              <div className="grid grid-cols-2 gap-4 border border-zinc-800 rounded-lg p-3 bg-zinc-900/50">
                <div>
                  <span className="text-zinc-400">Macro Timeframe:</span>{" "}
                  <span className="font-mono text-white">{config.macro_timeframe}</span>
                </div>
                <div>
                  <span className="text-zinc-400">Execution Timeframe:</span>{" "}
                  <span className="font-mono text-white">{config.execution_timeframe}</span>
                </div>
                <div>
                  <span className="text-zinc-400">Opening Range:</span>{" "}
                  <span className="font-mono text-white">
                    {config.execution_rules?.opening_range_start} -{" "}
                    {config.execution_rules?.opening_range_end} IST
                  </span>
                </div>
                <div>
                  <span className="text-zinc-400">Mandatory Square-off:</span>{" "}
                  <span className="font-mono text-rose-400 font-bold">
                    {config.execution_rules?.auto_square_off_time} IST
                  </span>
                </div>
              </div>

              <div className="border border-zinc-800 rounded-lg p-3 bg-zinc-900/50 space-y-2">
                <span className="font-semibold text-white">Confluence Gates:</span>
                <ul className="list-disc pl-5 space-y-1 text-zinc-400">
                  <li>Breakout above 15m Opening Range High with &gt;1.5x volume</li>
                  <li>Price sustaining above intraday VWAP</li>
                  <li>Price clearing CPR Top Central Pivot (TC)</li>
                  <li>Target 1: 1.8x ATR (books 50% profit, moves SL to entry)</li>
                  <li>Target 2: 3.0x ATR runner</li>
                  <li>Stop Loss: 1.2x ATR trailing risk protection</li>
                </ul>
              </div>

              <div className="flex justify-end pt-2">
                <button
                  onClick={() => setShowConfigModal(false)}
                  className="px-4 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-white rounded-lg text-xs"
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
