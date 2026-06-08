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
            const eventSource = new EventSource(url);

            eventSource.onmessage = (event) => {
                if (event.data === ': keep-alive' || !event.data) return;
                setLogs((prev) => [...prev, event.data]);
            };

            eventSource.onerror = () => {
                eventSource.close();
            };

            eventSourceRef.current = eventSource;
        } else {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
            setLogs([]);
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

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="bg-gray-900 rounded-xl shadow-2xl border border-gray-700 overflow-hidden flex flex-col w-full max-w-4xl h-[70vh] mx-4">
                <div className="bg-gray-800 px-6 py-3 flex items-center justify-between border-b border-gray-700">
                    <div className="flex items-center space-x-3">
                        <CommandLineIcon className="h-5 w-5 text-green-400" />
                        <span className="text-sm font-mono text-gray-300">Trading Cycle Console</span>
                        <span className="text-xs text-gray-500 font-mono">{logs.length} lines</span>
                    </div>
                    <div className="flex items-center space-x-3">
                        <button
                            onClick={() => setLogs([])}
                            className="text-gray-400 hover:text-white p-1.5 rounded hover:bg-gray-700 transition"
                            title="Clear"
                        >
                            <TrashIcon className="h-4 w-4" />
                        </button>
                        <button
                            onClick={onClose}
                            className="text-gray-400 hover:text-white p-1.5 rounded hover:bg-gray-700 transition"
                            title="Close"
                        >
                            <XMarkIcon className="h-5 w-5" />
                        </button>
                    </div>
                </div>

                <div
                    ref={scrollRef}
                    className="flex-1 p-4 font-mono text-xs text-green-400 bg-black overflow-y-auto"
                    style={{
                        scrollbarWidth: 'thin',
                        scrollbarColor: '#333 #000',
                    }}
                >
                    {logs.length === 0 ? (
                        <div className="text-gray-600 italic">Waiting for logs...</div>
                    ) : (
                        logs.map((log, index) => (
                            <div key={index} className="mb-1 break-words leading-relaxed">
                                <span className="text-gray-600 mr-2 select-none">[{index + 1}]</span>
                                {log}
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export default Terminal;
