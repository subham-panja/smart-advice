'use client';

import { useState } from 'react';
import { healthCheck, getRecommendations, triggerAnalysis } from '@/lib/api';

const ApiTest = () => {
  const [results, setResults] = useState<string[]>([]);

  const addResult = (message: string) => {
    setResults(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
  };

  const testHealthCheck = async () => {
    try {
      addResult('Testing health check...');
      const response = await healthCheck();
      addResult(`Health check success: ${JSON.stringify(response)}`);
    } catch (error) {
      addResult(`Health check error: ${error}`);
      console.error('Health check error:', error);
    }
  };

  const testRecommendations = async () => {
    try {
      addResult('Testing recommendations...');
      const response = await getRecommendations();
      addResult(`Recommendations success: ${JSON.stringify(response)}`);
    } catch (error) {
      addResult(`Recommendations error: ${error}`);
      console.error('Recommendations error:', error);
    }
  };

  const testTriggerAnalysis = async () => {
    try {
      addResult('Testing trigger analysis...');
      const response = await triggerAnalysis({ test: true, max_stocks: 1 });
      addResult(`Trigger analysis success: ${JSON.stringify(response)}`);
    } catch (error) {
      addResult(`Trigger analysis error: ${error}`);
      console.error('Trigger analysis error:', error);
    }
  };

  const clearResults = () => {
    setResults([]);
  };

  return (
    <div className="p-4 bg-gray-100 dark:bg-gray-900 rounded-lg transition-colors">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">API Connection Test</h3>
      
      <div className="space-x-2 mb-4">
        <button
          onClick={testHealthCheck}
          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Health Check
        </button>
        <button
          onClick={testRecommendations}
          className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
        >
          Test Recommendations
        </button>
        <button
          onClick={testTriggerAnalysis}
          className="px-3 py-1 bg-purple-500 text-white rounded hover:bg-purple-600"
        >
          Test Analysis
        </button>
        <button
          onClick={clearResults}
          className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
        >
          Clear
        </button>
      </div>

      <div className="text-sm">
        <p className="text-gray-900 dark:text-gray-100"><strong>API Base URL:</strong> {process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:5001'}</p>
      </div>

      <div className="mt-4 max-h-96 overflow-y-auto">
        <h4 className="font-medium mb-2 text-gray-900 dark:text-gray-100">Results:</h4>
        <div className="bg-white dark:bg-gray-800 p-3 rounded border border-gray-200 dark:border-gray-700 text-sm">
          {results.length === 0 ? (
            <p className="text-gray-500 dark:text-gray-400">No tests run yet.</p>
          ) : (
            results.map((result, index) => (
              <div key={index} className="mb-1 font-mono text-xs text-gray-900 dark:text-gray-100">
                {result}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default ApiTest;
