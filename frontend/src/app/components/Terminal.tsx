'use client';

import React, { useEffect, useRef, useState } from 'react';
import { XMarkIcon, CommandLineIcon, TrashIcon } from '@heroicons/react/24/outline';

interface TerminalProps {
    isOpen: boolean;
    onClose: () => void;
    apiHost: string;
}

const Terminal: React.FC<TerminalProps> = ({ isOpen, onClose, apiHost }) => {
    const [logs, setLogs] = useState<string[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);
    const eventSourceRef = useRef<EventSource | null>(null);

    useEffect(() => {
        if (isOpen) {
            const url = `${apiHost}/stream-logs`;
            console.log('Connecting to SSE:', url);
            const eventSource = new EventSource(url);

            eventSource.onmessage = (event) => {
                // Handle heartbeat/keep-alive
                if (event.data === ': keep-alive' || !event.data) return;
                setLogs((prev) => [...prev, event.data]);
            };

            eventSource.onerror = (error) => {
                console.error('SSE Error:', error);
                eventSource.close();
            };

            eventSourceRef.current = eventSource;
        } else {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
        }

        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }
        };
    }, [isOpen, apiHost]);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    const clearLogs = () => setLogs([]);

    if (!isOpen) return null;

    return (
        <div className="fixed bottom-4 right-4 w-full max-w-2xl z-50 animate-in slide-in-from-bottom-5 duration-300 px-4 sm:px-0">
            <div className="bg-gray-900 rounded-lg shadow-2xl border border-gray-700 overflow-hidden flex flex-col h-96">
                {/* Terminal Header */}
                <div className="bg-gray-800 px-4 py-2 flex items-center justify-between border-b border-gray-700">
                    <div className="flex items-center space-x-2">
                        <CommandLineIcon className="h-5 w-5 text-green-400" />
                        <span className="text-sm font-mono text-gray-300">Analysis Console</span>
                    </div>
                    <div className="flex items-center space-x-3">
                        <button
                            onClick={clearLogs}
                            className="text-gray-400 hover:text-white p-1 rounded hover:bg-gray-700 transition"
                            title="Clear Logs"
                        >
                            <TrashIcon className="h-4 w-4" />
                        </button>
                        <button
                            onClick={onClose}
                            className="text-gray-400 hover:text-white p-1 rounded hover:bg-gray-700 transition"
                            title="Close"
                        >
                            <XMarkIcon className="h-5 w-5" />
                        </button>
                    </div>
                </div>

                {/* Terminal Content */}
                <div
                    ref={scrollRef}
                    className="flex-1 p-4 font-mono text-[10px] sm:text-xs text-green-400 bg-black overflow-y-auto custom-scrollbar"
                >
                    {logs.length === 0 ? (
                        <div className="text-gray-600 italic">Waiting for analysis logs...</div>
                    ) : (
                        logs.map((log, index) => (
                            <div key={index} className="mb-1 break-words">
                                <span className="text-gray-600 mr-2 opacity-50select-none">[{index + 1}]</span>
                                {log}
                            </div>
                        ))
                    )}
                </div>
            </div>

            {/* Scrollbar Styling */}
            <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #000;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #333;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #444;
        }
      `}</style>
        </div>
    );
};

export default Terminal;
