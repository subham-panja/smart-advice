"use client";

import { useState, useEffect, useMemo } from "react";
import { getRecommendations, updateRecommendation, StockRecommendation } from "@/lib/api";
import { ArrowPathIcon } from "@heroicons/react/24/outline";
import { ColumnDef } from "@tanstack/react-table";
import DataTable from "../components/DataTable";
import StockPickNoteTag from "../components/StockPickNoteTag";
import { cleanSymbol, formatDecimal } from "@/lib/utils";

export default function RecommendationsPage() {
  const [recommendations, setRecommendations] = useState<StockRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    async function fetchRecommendations() {
      try {
        const response = await getRecommendations();

        if (response.status === "success") {
          setRecommendations(response.recommendations || []);
        } else {
          setError(response.error || "Failed to load recommendations.");
        }
      } catch (error) {
        console.error("Error fetching recommendations:", error);
        setError("Failed to connect to server.");
      } finally {
        setLoading(false);
      }
    }

    fetchRecommendations();
  }, []);

  const handleRefresh = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getRecommendations();

      if (response.status === "success") {
        setRecommendations(response.recommendations || []);
      } else {
        setError(response.error || "Failed to load recommendations.");
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error);
      setError("Failed to connect to server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-6 gap-4">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Stock Recommendations</h1>
        <button
          onClick={handleRefresh}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          <ArrowPathIcon className="h-5 w-5" />
          <span>Refresh</span>
        </button>
      </div>

      {loading ? (
        <div className="text-center text-gray-500">Loading recommendations...</div>
      ) : error ? (
        <div className="text-center text-red-500">{error}</div>
      ) : recommendations.length === 0 ? (
        <div className="text-center text-gray-500">No recommendations available.</div>
      ) : (
        <div>
          <DataTable
            columns={columns}
            data={recommendations}
            searchPlaceholder="Search recommendations..."
            pageSize={15}
          />
        </div>
      )}
    </div>
  );
}

