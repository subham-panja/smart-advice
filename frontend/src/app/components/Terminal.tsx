'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { XMarkIcon, CommandLineIcon, TrashIcon } from '@heroicons/react/24/outline';

interface PendingExit {
    symbol: string;
    systemPrice: number;
    exitReason: string;
    quantity: number;
    entryPrice: number;
    pnlPct: number;
    confirmed: boolean;
    confirming: boolean;
}

interface TerminalProps {
    isOpen: boolean;
    onClose: () => void;
    apiHost: string;
}

const Terminal: React.FC<TerminalProps> = ({ isOpen, onClose, apiHost }) => {
    const [logs, setLogs] = useState<string[]>([]);
    const [pendingExits, setPendingExits] = useState<PendingExit[]>([]);
    const [editPrices, setEditPrices] = useState<Record<string, string>>({});
    const scrollRef = useRef<HTMLDivElement>(null);
    const eventSourceRef = useRef<EventSource | null>(null);

    const parseExitConfirmEvent = useCallback((data: string): PendingExit | null => {
        // Format: EXIT_CONFIRM:SYMBOL:PRICE:REASON:QTY:ENTRY_PRICE:PNL_PCT
        const prefix = 'IMPORTANT | EXIT_CONFIRM:';
        if (!data.startsWith(prefix)) return null;

        const parts = data.substring(prefix.length).split(':');
        if (parts.length < 6) return null;

        return {
            symbol: parts[0],
            systemPrice: parseFloat(parts[1]),
            exitReason: parts[2],
            quantity: parseInt(parts[3]),
            entryPrice: parseFloat(parts[4]),
            pnlPct: parseFloat(parts[5]),
            confirmed: false,
            confirming: false,
        };
    }, []);

    const handleConfirmExit = useCallback(async (symbol: string, price: number) => {
        setPendingExits(prev =>
            prev.map(e => e.symbol === symbol ? { ...e, confirming: true } : e)
        );

        try {
            const res = await fetch(`${apiHost}/confirm-exit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ symbol, exit_price: price }),
            });

            const data = await res.json();

            if (data.status === 'success') {
                setPendingExits(prev =>
                    prev.map(e => e.symbol === symbol ? { ...e, confirmed: true, confirming: false } : e)
                );
                setLogs(prev => {
                    const isBtst = data.data?.settlement?.is_btst;
                    const settleMsg = isBtst
                        ? `0% Available (BTST) | 100% settles ${data.data?.settlement?.settlement_date || 'T+1'}`
                        : `80% (₹${data.data?.settlement?.immediate?.toFixed(2) || '?'}) available now | 20% settles ${data.data?.settlement?.settlement_date || 'T+1'}`;

                    return [
                        ...prev,
                        `✅ ${symbol} exit confirmed @ ₹${price.toFixed(2)} | ${settleMsg}`
                    ];
                });
            } else {
                setPendingExits(prev =>
                    prev.map(e => e.symbol === symbol ? { ...e, confirming: false } : e)
                );
                setLogs(prev => [...prev, `❌ Failed to confirm ${symbol}: ${data.error}`]);
            }
        } catch (err) {
            setPendingExits(prev =>
                prev.map(e => e.symbol === symbol ? { ...e, confirming: false } : e)
            );
            setLogs(prev => [...prev, `❌ Network error confirming ${symbol}: ${err}`]);
        }
    }, [apiHost]);

    // Fetch any existing pending exits when terminal opens
    useEffect(() => {
        if (isOpen) {
            fetch(`${apiHost}/pending-exits`)
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'success' && data.pending_exits?.length > 0) {
                        const exits: PendingExit[] = data.pending_exits.map((p: Record<string, unknown>) => ({
                            symbol: p.symbol as string,
                            systemPrice: p.system_exit_price as number,
                            exitReason: p.exit_reason as string,
                            quantity: p.quantity as number,
                            entryPrice: p.entry_price as number,
                            pnlPct: p.pnl_pct as number,
                            confirmed: false,
                            confirming: false,
                        }));
                        setPendingExits(prev => {
                            const existingSymbols = new Set(prev.map(e => e.symbol));
                            const newExits = exits.filter((e: PendingExit) => !existingSymbols.has(e.symbol));
                            return [...prev, ...newExits];
                        });
                    }
                })
                .catch(() => { /* silent */ });
        }
    }, [isOpen, apiHost]);

    useEffect(() => {
        if (isOpen) {
            const url = `${apiHost}/stream-logs`;
            const eventSource = new EventSource(url);

            eventSource.onmessage = (event) => {
                if (event.data === ': keep-alive' || !event.data) return;

                // Check if this is an exit confirmation event
                const exitEvent = parseExitConfirmEvent(event.data);
                if (exitEvent) {
                    setPendingExits(prev => {
                        if (prev.some(e => e.symbol === exitEvent.symbol)) return prev;
                        return [...prev, exitEvent];
                    });
                    // Don't add the raw EXIT_CONFIRM protocol message to visible logs
                    return;
                }

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
            setPendingExits([]);
            setEditPrices({});
        }

        return () => {
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
            }
        };
    }, [isOpen, apiHost, parseExitConfirmEvent]);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs, pendingExits]);

    if (!isOpen) return null;

    const activePending = pendingExits.filter(e => !e.confirmed);

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="bg-gray-900 rounded-xl shadow-2xl border border-gray-700 overflow-hidden flex flex-col w-full max-w-4xl h-[70vh] mx-4">
                <div className="bg-gray-800 px-6 py-3 flex items-center justify-between border-b border-gray-700">
                    <div className="flex items-center space-x-3">
                        <CommandLineIcon className="h-5 w-5 text-green-400" />
                        <span className="text-sm font-mono text-gray-300">Trading Cycle Console</span>
                        <span className="text-xs text-gray-500 font-mono">{logs.length} lines</span>
                        {activePending.length > 0 && (
                            <span className="text-xs bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded-full font-mono animate-pulse">
                                {activePending.length} exit{activePending.length > 1 ? 's' : ''} pending
                            </span>
                        )}
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

                {/* Pending Exit Confirmation Cards */}
                {activePending.length > 0 && (
                    <div className="border-b border-gray-700 bg-gray-800/50 px-4 py-3 space-y-3 max-h-[30vh] overflow-y-auto">
                        {activePending.map((exit) => (
                            <div
                                key={exit.symbol}
                                className="bg-gray-900/80 border border-amber-500/30 rounded-lg p-4"
                            >
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center space-x-2">
                                        <span className="text-amber-400 text-lg">🛑</span>
                                        <span className="text-white font-bold font-mono">{exit.symbol}</span>
                                        <span className={`text-xs px-2 py-0.5 rounded ${exit.pnlPct >= 0 ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                                            {exit.pnlPct >= 0 ? '+' : ''}{exit.pnlPct.toFixed(2)}%
                                        </span>
                                    </div>
                                    <span className="text-xs text-gray-400 font-mono">{exit.exitReason}</span>
                                </div>

                                <div className="grid grid-cols-3 gap-3 text-xs text-gray-400 mb-3">
                                    <div>
                                        <span className="block text-gray-500">Entry</span>
                                        <span className="text-gray-300 font-mono">₹{exit.entryPrice.toFixed(2)}</span>
                                    </div>
                                    <div>
                                        <span className="block text-gray-500">System Exit</span>
                                        <span className="text-white font-mono font-bold">₹{exit.systemPrice.toFixed(2)}</span>
                                    </div>
                                    <div>
                                        <span className="block text-gray-500">Qty</span>
                                        <span className="text-gray-300 font-mono">{exit.quantity}</span>
                                    </div>
                                </div>

                                <div className="text-xs text-amber-400/80 mb-3">
                                    Did you exit at ₹{exit.systemPrice.toFixed(2)}? Confirm or enter the actual price:
                                </div>

                                <div className="flex items-center space-x-2">
                                    <button
                                        onClick={() => handleConfirmExit(exit.symbol, exit.systemPrice)}
                                        disabled={exit.confirming}
                                        className="px-3 py-1.5 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 text-white text-xs font-medium rounded transition"
                                    >
                                        {exit.confirming ? 'Confirming...' : `✓ Confirm ₹${exit.systemPrice.toFixed(2)}`}
                                    </button>

                                    <div className="flex items-center space-x-1 flex-1">
                                        <input
                                            type="number"
                                            step="0.01"
                                            placeholder="Actual price"
                                            value={editPrices[exit.symbol] || ''}
                                            onChange={(e) =>
                                                setEditPrices(prev => ({ ...prev, [exit.symbol]: e.target.value }))
                                            }
                                            className="flex-1 bg-gray-800 border border-gray-600 rounded px-2 py-1.5 text-xs text-white font-mono placeholder-gray-500 focus:border-amber-500 focus:outline-none"
                                        />
                                        <button
                                            onClick={() => {
                                                const price = parseFloat(editPrices[exit.symbol]);
                                                if (!isNaN(price) && price > 0) {
                                                    handleConfirmExit(exit.symbol, price);
                                                }
                                            }}
                                            disabled={exit.confirming || !editPrices[exit.symbol]}
                                            className="px-3 py-1.5 bg-amber-600 hover:bg-amber-500 disabled:bg-gray-600 text-white text-xs font-medium rounded transition"
                                        >
                                            Submit
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* Confirmed exits (brief green flash) */}
                {pendingExits.filter(e => e.confirmed).length > 0 && (
                    <div className="border-b border-gray-700 bg-green-900/20 px-4 py-2">
                        {pendingExits.filter(e => e.confirmed).map((exit) => (
                            <div key={`done-${exit.symbol}`} className="text-xs text-green-400 font-mono">
                                ✅ {exit.symbol} exit confirmed — processing settlement...
                            </div>
                        ))}
                    </div>
                )}

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
