"use client";

import { useState, useEffect, useMemo } from "react";
import { getRecommendations, StockRecommendation } from "@/lib/api";
import { ArrowPathIcon } from "@heroicons/react/24/outline";
import { ColumnDef } from "@tanstack/react-table";
import DataTable from "../components/DataTable";

export default function RecommendationsPage() {
  const [recommendations, setRecommendations] = useState<StockRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const columns = useMemo<ColumnDef<StockRecommendation>[]>(() => [
    {
      header: "Symbol",
      accessorKey: "symbol",
    },
    {
      header: "Company Name",
      accessorKey: "company_name",
    },
    {
      header: "Technical Score",
      accessorKey: "technical_score",
    },
    {
      header: "Fundamental Score",
      accessorKey: "fundamental_score",
    },
    {
      header: "Sentiment Score",
      accessorKey: "sentiment_score",
    },
    {
      header: "Combined Score",
      accessorKey: "combined_score",
    },
    {
      header: "Recommendation",
      accessorKey: "recommendation_strength",
    },
    {
      header: "Backtest CAGR",
      accessorKey: "backtest_cagr",
      cell: ({ row }) => row.original.backtest_cagr ?? "-",
    },
    {
      header: "Date",
      accessorKey: "recommendation_date",
      cell: ({ row }) => new Date(row.original.recommendation_date).toLocaleDateString(),
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
    <div className="w-full max-w-7xl mx-auto px-0">
      <div className="flex items-center justify-between mb-6">
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

