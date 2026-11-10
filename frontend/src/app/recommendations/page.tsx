"use client";

import { useState, useEffect, useMemo } from "react";
import { getRecommendations, updateRecommendation, StockRecommendation } from "@/lib/api";
import {
  ArrowPathIcon,
  MagnifyingGlassIcon,
  SparklesIcon,
  FunnelIcon,
  ArrowTrendingUpIcon,
  ShieldCheckIcon,
} from "@heroicons/react/24/outline";
import { ColumnDef } from "@tanstack/react-table";
import DataTable from "../components/DataTable";
import StockPickNoteTag from "../components/StockPickNoteTag";
import { cleanSymbol, formatDecimal } from "@/lib/utils";

export default function RecommendationsPage() {
  const [recommendations, setRecommendations] = useState<StockRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filter & Search states
  const [searchQuery, setSearchQuery] = useState("");
  const [strengthFilter, setStrengthFilter] = useState<string>("ALL");
  const [sortBy, setSortBy] = useState<"score" | "return" | "symbol">("score");

  const columns = useMemo<ColumnDef<StockRecommendation>[]>(() => [
    {
      header: "Symbol",
      accessorKey: "symbol",
      cell: ({ row }) => (
        <span className="font-semibold text-gray-900 dark:text-gray-100">
          {cleanSymbol(row.original.symbol)}
        </span>
      ),
    },
    {
      header: "Company Name",
      accessorKey: "company_name",
    },
    {
      header: "Why Picked (Setup Context)",
      id: "why_picked",
      cell: ({ row }) => (
        <StockPickNoteTag
          symbol={cleanSymbol(row.original.symbol)}
          strategy={row.original.strategy_name || "Swing_Trading_v3"}
          reason={row.original.reason}
          note={row.original.note}
          technicalScore={row.original.technical_score}
          fundamentalScore={row.original.fundamental_score}
          sentimentScore={row.original.sentiment_score}
          combinedScore={row.original.combined_score}
          compositeScore={row.original.composite_score}
          entryPrice={row.original.buy_price}
          stopLoss={row.original.stop_loss || (row.original.buy_price ? row.original.buy_price * 0.94 : undefined)}
          target={row.original.sell_price}
          entryAtr={row.original.entry_atr}
          rrRatio={row.original.rr_ratio || 3.0}
          suggestedQuantity={row.original.suggested_quantity}
          allocationPct={row.original.allocation_pct}
          recommendationStrength={row.original.recommendation_strength}
          patternDetected={row.original.pattern_detected}
          setupChecklist={row.original.setup_checklist}
          technicalPillars={row.original.technical_pillars}
          recommendation={row.original}
          onSaveNote={async (newNote) => {
            await updateRecommendation(row.original.symbol, { note: newNote });
            handleRefresh();
          }}
        />
      ),
    },
    {
      header: "Technical Score",
      accessorKey: "technical_score",
      cell: ({ row }) => (
        <span className="font-mono text-xs font-semibold text-gray-700 dark:text-gray-300">
          {formatDecimal(row.original.technical_score, 2, false)}
        </span>
      ),
    },
    {
      header: "Fundamental Score",
      accessorKey: "fundamental_score",
      cell: ({ row }) => (
        <span className="font-mono text-xs text-gray-500 dark:text-gray-400">
          {formatDecimal(row.original.fundamental_score, 2, true)}
        </span>
      ),
    },
    {
      header: "Sentiment Score",
      accessorKey: "sentiment_score",
      cell: ({ row }) => (
        <span className="font-mono text-xs text-gray-500 dark:text-gray-400">
          {formatDecimal(row.original.sentiment_score, 2, true)}
        </span>
      ),
    },
    {
      header: "Combined Score",
      accessorKey: "combined_score",
      cell: ({ row }) => (
        <span className="font-mono text-xs font-bold text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/30 px-2 py-0.5 rounded">
          {formatDecimal(row.original.combined_score, 2, false)}
        </span>
      ),
    },
    {
      header: "Recommendation",
      accessorKey: "recommendation_strength",
      cell: ({ row }) => (
        <span className="px-2.5 py-1 text-xs font-bold rounded-full bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">
          {row.original.recommendation_strength || "BUY"}
        </span>
      ),
    },
    {
      header: "Backtest CAGR",
      accessorKey: "backtest_cagr",
      cell: ({ row }) =>
        row.original.backtest_cagr && Number(row.original.backtest_cagr) !== 0 ? (
          <span className="font-mono text-xs font-semibold text-emerald-600 dark:text-emerald-400">
            +{formatDecimal(row.original.backtest_cagr, 2, true)}%
          </span>
        ) : (
          <span className="text-gray-400">-</span>
        ),
    },
    {
      header: "Date",
      accessorKey: "recommendation_date",
      cell: ({ row }) => (
        <span className="text-xs text-gray-500 dark:text-gray-400">
          {row.original.recommendation_date ? new Date(row.original.recommendation_date).toLocaleDateString("en-IN", { month: "short", day: "numeric", year: "numeric" }) : "-"}
        </span>
      ),
    },
  ], []);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getRecommendations();

      if (response && response.status === "success") {
        setRecommendations(response.recommendations || []);
      } else {
        setError(response?.error || "Failed to load recommendations.");
      }
    } catch (err) {
      console.error("Error fetching recommendations:", err);
      setError("Failed to connect to server.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
  }, []);

  const handleRefresh = async () => {
    await fetchRecommendations();
  };

  // Filter and sort recommendations
  const filteredRecommendations = useMemo(() => {
    let result = [...recommendations];

    // Search query filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase().trim();
      result = result.filter(
        (r) =>
          cleanSymbol(r.symbol).toLowerCase().includes(query) ||
          (r.company_name && r.company_name.toLowerCase().includes(query)) ||
          (r.pattern_detected && r.pattern_detected.toLowerCase().includes(query))
      );
    }

    // Recommendation strength filter
    if (strengthFilter !== "ALL") {
      result = result.filter((r) => {
        const strength = (r.recommendation_strength || "BUY").toUpperCase();
        if (strengthFilter === "STRONG BUY") return strength.includes("STRONG");
        if (strengthFilter === "BUY") return strength === "BUY";
        return strength === strengthFilter;
      });
    }

    // Sorting
    result.sort((a, b) => {
      if (sortBy === "score") {
        const scoreA = a.composite_score ?? a.combined_score ?? 0;
        const scoreB = b.composite_score ?? b.combined_score ?? 0;
        return scoreB - scoreA;
      }
      if (sortBy === "return") {
        const retA =
          a.expected_return_percent ??
          (a.buy_price && a.sell_price ? ((a.sell_price - a.buy_price) / a.buy_price) * 100 : 0);
        const retB =
          b.expected_return_percent ??
          (b.buy_price && b.sell_price ? ((b.sell_price - b.buy_price) / b.buy_price) * 100 : 0);
        return retB - retA;
      }
      if (sortBy === "symbol") {
        return cleanSymbol(a.symbol).localeCompare(cleanSymbol(b.symbol));
      }
      return 0;
    });

    return result;
  }, [recommendations, searchQuery, strengthFilter, sortBy]);

  // Summary Metrics
  const strongBuysCount = useMemo(
    () => recommendations.filter((r) => (r.recommendation_strength || "").toUpperCase().includes("STRONG")).length,
    [recommendations]
  );

  const avgComposite = useMemo(() => {
    if (recommendations.length === 0) return 0;
    const sum = recommendations.reduce((acc, r) => acc + (r.composite_score ?? r.combined_score ?? 0), 0);
    return sum / recommendations.length;
  }, [recommendations]);

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
      {/* Modern Glassmorphic Header */}
      <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-md border border-gray-200/80 dark:border-gray-700/80 shadow-sm rounded-2xl p-5 sm:p-6 space-y-5">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-2.5">
              <span>Stock Recommendations</span>
              <span className="hidden sm:inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300">
                <SparklesIcon className="h-3.5 w-3.5" />
                AI Powered
              </span>
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              High-conviction algorithmic setups, pattern recognition & multi-factor scores
            </p>
          </div>
          <button
            onClick={handleRefresh}
            disabled={loading}
            className="inline-flex items-center gap-2 px-4 py-2.5 bg-blue-600 text-white rounded-xl hover:bg-blue-700 font-medium text-sm transition shadow-sm active:scale-95 disabled:opacity-50"
          >
            <ArrowPathIcon className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
        </div>

        {/* Quick Stats Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
          <div className="bg-gray-50/70 dark:bg-gray-900/50 p-3 rounded-xl border border-gray-100 dark:border-gray-800">
            <span className="text-[10px] uppercase font-semibold text-gray-400 dark:text-gray-500 block tracking-wider">
              Total Recommendations
            </span>
            <span className="text-xl font-bold text-gray-900 dark:text-gray-100">
              {recommendations.length}
            </span>
          </div>
          <div className="bg-gray-50/70 dark:bg-gray-900/50 p-3 rounded-xl border border-gray-100 dark:border-gray-800">
            <span className="text-[10px] uppercase font-semibold text-gray-400 dark:text-gray-500 block tracking-wider">
              Strong Buys
            </span>
            <span className="text-xl font-bold text-emerald-600 dark:text-emerald-400">
              {strongBuysCount}
            </span>
          </div>
          <div className="bg-gray-50/70 dark:bg-gray-900/50 p-3 rounded-xl border border-gray-100 dark:border-gray-800">
            <span className="text-[10px] uppercase font-semibold text-gray-400 dark:text-gray-500 block tracking-wider">
              Avg Composite
            </span>
            <span className="text-xl font-bold text-blue-600 dark:text-blue-400">
              {avgComposite > 0 ? formatDecimal(avgComposite, 1, false) : "—"}
            </span>
          </div>
          <div className="bg-gray-50/70 dark:bg-gray-900/50 p-3 rounded-xl border border-gray-100 dark:border-gray-800">
            <span className="text-[10px] uppercase font-semibold text-gray-400 dark:text-gray-500 block tracking-wider">
              Filtered Picks
            </span>
            <span className="text-xl font-bold text-gray-900 dark:text-gray-100">
              {filteredRecommendations.length}
            </span>
          </div>
        </div>

        {/* Filter & Search Bar */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-2 border-t border-gray-100 dark:border-gray-700/60">
          <div className="relative flex-1 max-w-md">
            <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none">
              <MagnifyingGlassIcon className="h-4 w-4 text-gray-400" />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search symbol, company, or setup pattern..."
              className="w-full pl-10 pr-4 py-2 text-sm bg-gray-50/90 dark:bg-gray-900/80 border border-gray-200 dark:border-gray-700 rounded-xl text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
            />
          </div>

          <div className="flex items-center gap-2 overflow-x-auto pb-1 sm:pb-0">
            {/* Strength Filter Pills */}
            <div className="flex items-center bg-gray-100 dark:bg-gray-900/80 p-1 rounded-xl text-xs font-medium">
              {(["ALL", "STRONG BUY", "BUY"] as const).map((strength) => (
                <button
                  key={strength}
                  onClick={() => setStrengthFilter(strength)}
                  className={`px-3 py-1.5 rounded-lg transition ${
                    strengthFilter === strength
                      ? "bg-white dark:bg-gray-800 text-blue-600 dark:text-blue-400 shadow-xs font-semibold"
                      : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200"
                  }`}
                >
                  {strength}
                </button>
              ))}
            </div>

            {/* Sort Selector */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-3 py-2 text-xs font-medium bg-gray-100 dark:bg-gray-900/80 border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="score">Sort: Score</option>
              <option value="return">Sort: Expected Return</option>
              <option value="symbol">Sort: Symbol</option>
            </select>
          </div>
        </div>
      </div>

      {/* Content Area */}
      {loading ? (
        <div className="p-12 text-center text-gray-500">
          <ArrowPathIcon className="h-8 w-8 mx-auto text-blue-500 animate-spin mb-3" />
          <p className="text-sm font-medium">Loading recommendations...</p>
        </div>
      ) : error ? (
        <div className="p-8 text-center text-red-500 bg-red-50 dark:bg-red-950/20 rounded-2xl border border-red-200 dark:border-red-800">
          <p className="font-semibold">{error}</p>
          <button
            onClick={handleRefresh}
            className="mt-3 px-4 py-2 text-xs font-medium bg-red-600 text-white rounded-lg hover:bg-red-700 transition"
          >
            Retry
          </button>
        </div>
      ) : recommendations.length === 0 ? (
        <div className="p-16 text-center bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700">
          <ArrowTrendingUpIcon className="h-12 w-12 mx-auto text-gray-300 dark:text-gray-600 mb-3" />
          <p className="text-gray-500 dark:text-gray-400 text-sm font-medium">
            No recommendations available.
          </p>
        </div>
      ) : filteredRecommendations.length === 0 ? (
        <div className="p-12 text-center bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700">
          <p className="text-gray-500 dark:text-gray-400 text-sm font-medium">
            No recommendations match your search or filter.
          </p>
          <button
            onClick={() => {
              setSearchQuery("");
              setStrengthFilter("ALL");
            }}
            className="mt-3 px-4 py-2 text-xs font-medium text-blue-600 dark:text-blue-400 hover:underline"
          >
            Clear filters
          </button>
        </div>
      ) : (
        <div>
          {/* Desktop Rich DataTable (>=md) */}
          <div className="hidden md:block">
            <DataTable
              columns={columns}
              data={filteredRecommendations}
              searchPlaceholder="Search recommendations..."
              pageSize={15}
            />
          </div>

          {/* Mobile Touch-Friendly Card List (<md) */}
          <div className="block md:hidden space-y-4">
            {filteredRecommendations.map((rec) => {
              const cleanSym = cleanSymbol(rec.symbol);
              const avatarInitials = cleanSym.slice(0, 2).toUpperCase();
              const sl = rec.stop_loss || (rec.buy_price ? rec.buy_price * 0.94 : undefined);
              const expectedReturn =
                rec.expected_return_percent ??
                (rec.buy_price && rec.sell_price
                  ? ((rec.sell_price - rec.buy_price) / rec.buy_price) * 100
                  : 0);
              const compScore = rec.composite_score ?? rec.combined_score;
              const strength = (rec.recommendation_strength || "BUY").toUpperCase();
              const isStrong = strength.includes("STRONG");

              return (
                <div
                  key={rec.symbol}
                  className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-200 dark:border-gray-700 p-4 shadow-sm space-y-3.5 hover:border-blue-400 dark:hover:border-blue-500 transition-colors"
                >
                  {/* Top Header: Avatar, Symbol, Company & Strength Badge */}
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-3 min-w-0">
                      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/10 via-teal-500/10 to-blue-500/10 dark:from-emerald-500/20 dark:to-blue-500/20 border border-emerald-200/50 dark:border-emerald-700/40 flex items-center justify-center font-bold text-sm text-emerald-700 dark:text-emerald-300 flex-shrink-0">
                        {avatarInitials}
                      </div>
                      <div className="min-w-0">
                        <span className="font-bold text-base text-gray-900 dark:text-gray-100 block">
                          {cleanSym}
                        </span>
                        <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                          {rec.company_name}
                        </p>
                      </div>
                    </div>

                    <span
                      className={`px-2.5 py-1 text-xs font-bold rounded-full flex-shrink-0 border ${
                        isStrong
                          ? "bg-emerald-100 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 border-emerald-300 dark:border-emerald-800"
                          : "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border-green-200 dark:border-green-800/50"
                      }`}
                    >
                      {rec.recommendation_strength || "BUY"}
                    </span>
                  </div>

                  {/* Why Picked Setup Context Badge */}
                  <div className="pt-0.5">
                    <StockPickNoteTag
                      symbol={cleanSym}
                      strategy={rec.strategy_name || "Swing_Trading_v3"}
                      reason={rec.reason}
                      note={rec.note}
                      technicalScore={rec.technical_score}
                      fundamentalScore={rec.fundamental_score}
                      sentimentScore={rec.sentiment_score}
                      combinedScore={rec.combined_score}
                      compositeScore={rec.composite_score}
                      entryPrice={rec.buy_price}
                      stopLoss={sl}
                      target={rec.sell_price}
                      entryAtr={rec.entry_atr}
                      rrRatio={rec.rr_ratio || 3.0}
                      suggestedQuantity={rec.suggested_quantity}
                      allocationPct={rec.allocation_pct}
                      recommendationStrength={rec.recommendation_strength}
                      patternDetected={rec.pattern_detected}
                      setupChecklist={rec.setup_checklist}
                      technicalPillars={rec.technical_pillars}
                      recommendation={rec}
                      onSaveNote={async (newNote) => {
                        await updateRecommendation(rec.symbol, { note: newNote });
                        handleRefresh();
                      }}
                    />
                  </div>

                  {/* Score Pills */}
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg font-mono font-semibold bg-blue-50 dark:bg-blue-950/40 text-blue-700 dark:text-blue-300 border border-blue-200/60 dark:border-blue-800/60">
                      <span>Composite:</span>
                      <span>{compScore != null ? formatDecimal(compScore, 1, false) : "—"}</span>
                    </span>

                    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg font-mono font-medium bg-gray-50 dark:bg-gray-900/60 text-gray-700 dark:text-gray-300 border border-gray-200/60 dark:border-gray-800/60">
                      <span>Tech:</span>
                      <span>{rec.technical_score != null ? formatDecimal(rec.technical_score, 1, false) : "—"}</span>
                    </span>

                    {rec.backtest_cagr != null && Number(rec.backtest_cagr) !== 0 && (
                      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg font-mono font-medium bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400 border border-emerald-200/60 dark:border-emerald-800/60">
                        <span>CAGR:</span>
                        <span>+{formatDecimal(rec.backtest_cagr, 1, true)}%</span>
                      </span>
                    )}
                  </div>

                  {/* Pricing & Expected Return 2x2 Grid */}
                  <div className="grid grid-cols-2 gap-2 bg-gray-50/80 dark:bg-gray-900/60 p-3 rounded-xl border border-gray-100 dark:border-gray-800 text-xs">
                    <div>
                      <span className="text-[10px] uppercase font-semibold text-gray-400 dark:text-gray-500 block tracking-wider">
                        Buy Price
                      </span>
                      <span className="font-semibold text-gray-900 dark:text-gray-100">
                        {rec.buy_price ? `₹${formatDecimal(rec.buy_price, 2, false)}` : "—"}
                      </span>
                    </div>

                    <div>
                      <span className="text-[10px] uppercase font-semibold text-gray-400 dark:text-gray-500 block tracking-wider">
                        Target Price
                      </span>
                      <span className="font-semibold text-green-600 dark:text-green-400">
                        {rec.sell_price ? `₹${formatDecimal(rec.sell_price, 2, false)}` : "—"}
                      </span>
                    </div>

                    <div>
                      <span className="text-[10px] uppercase font-semibold text-gray-400 dark:text-gray-500 block tracking-wider">
                        Stop Loss
                      </span>
                      <span className="font-semibold text-red-600 dark:text-red-400">
                        {sl ? `₹${formatDecimal(sl, 2, false)}` : "—"}
                      </span>
                    </div>

                    <div>
                      <span className="text-[10px] uppercase font-semibold text-gray-400 dark:text-gray-500 block tracking-wider">
                        Expected Return
                      </span>
                      <span className="font-bold text-emerald-600 dark:text-emerald-400">
                        {expectedReturn > 0
                          ? `+${expectedReturn.toFixed(1)}%`
                          : expectedReturn !== 0
                          ? `${expectedReturn.toFixed(1)}%`
                          : "—"}
                      </span>
                    </div>
                  </div>

                  {/* Sizing & Date Footer */}
                  <div className="flex items-center justify-between text-[11px] text-gray-500 dark:text-gray-400 pt-1 border-t border-gray-100 dark:border-gray-700/60">
                    <div>
                      {rec.suggested_quantity ? (
                        <span>
                          Sizing:{" "}
                          <strong className="text-gray-800 dark:text-gray-200 font-semibold">
                            {rec.suggested_quantity} shares
                          </strong>
                          {rec.allocation_pct ? ` (${rec.allocation_pct}%)` : ""}
                        </span>
                      ) : (
                        <span>Risk/Reward: {rec.rr_ratio || 3.0}R</span>
                      )}
                    </div>
                    <span>
                      {rec.recommendation_date
                        ? new Date(rec.recommendation_date).toLocaleDateString("en-IN", {
                            month: "short",
                            day: "numeric",
                          })
                        : ""}
                    </span>
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
