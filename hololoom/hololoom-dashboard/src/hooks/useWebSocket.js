import { useEffect, useState, useRef, useCallback } from 'react';

export const useWebSocket = (url) => {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState(null);
    const [data, setData] = useState({
        analytics: null,
        skills: null,
        recentExecutions: []
    });

    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);

    const connect = useCallback(() => {
        try {
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket Connected');
                setIsConnected(true);
                // Clear any reconnect timeout if we connected successfully
                if (reconnectTimeoutRef.current) {
                    clearTimeout(reconnectTimeoutRef.current);
                    reconnectTimeoutRef.current = null;
                }
            };

            ws.onclose = () => {
                console.log('WebSocket Disconnected');
                setIsConnected(false);
                // Attempt reconnect after 3 seconds
                reconnectTimeoutRef.current = setTimeout(() => {
                    connect();
                }, 3000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket Error:', error);
                ws.close();
            };

            ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    setLastMessage(message);

                    if (message.type === 'initial') {
                        setData({
                            analytics: message.analytics,
                            skills: message.skills,
                            recentExecutions: message.recent_executions
                        });
                    } else if (message.type === 'analytics_update') {
                        setData(prev => ({
                            ...prev,
                            analytics: message.data
                        }));
                    }
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e);
                }
            };

        } catch (error) {
            console.error('Connection failed:', error);
        }
    }, [url]);

    useEffect(() => {
        connect();

        // Ping interval to keep connection alive
        const pingInterval = setInterval(() => {
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);

        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            clearInterval(pingInterval);
        };
    }, [connect]);

    return { isConnected, lastMessage, data };
};
