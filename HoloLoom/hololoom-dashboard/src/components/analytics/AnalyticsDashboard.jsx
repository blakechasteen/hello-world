import React from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    AreaChart, Area
} from 'recharts';
import { GlassPanel } from '../ui/Glass';

const MetricCard = ({ label, value, subtext, trend }) => (
    <GlassPanel intensity="low" className="metric-card" style={{ padding: '1.5rem' }}>
        <div style={{ color: 'var(--color-text-secondary)', fontSize: '0.9rem', marginBottom: '0.5rem' }}>
            {label}
        </div>
        <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--color-text-primary)' }}>
            {value}
        </div>
        {subtext && (
            <div style={{
                fontSize: '0.8rem',
                marginTop: '0.5rem',
                color: trend === 'up' ? 'var(--color-accent-primary)' : 'var(--color-text-muted)'
            }}>
                {subtext}
            </div>
        )}
    </GlassPanel>
);

const AnalyticsDashboard = ({ data }) => {
    if (!data || !data.analytics) {
        return (
            <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--color-text-muted)' }}>
                Waiting for analytics stream...
            </div>
        );
    }

    const { analytics, recentExecutions } = data;

    // Prepare chart data from recent executions
    const chartData = (recentExecutions || [])
        .slice()
        .reverse()
        .map(exec => ({
            time: new Date(exec.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            quality: (exec.quality_gain || 0) * 100,
            iterations: exec.iterations
        }));

    return (
        <div className="analytics-dashboard" style={{ padding: '1.5rem', height: '100%', overflowY: 'auto' }}>
            {/* Top Metrics Grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '1.5rem',
                marginBottom: '2rem'
            }}>
                <MetricCard
                    label="Total Queries"
                    value={analytics.total_queries || 0}
                    subtext="Lifetime executions"
                />
                <MetricCard
                    label="Avg Quality Gain"
                    value={`${((analytics.avg_quality_gain || 0) * 100).toFixed(1)}%`}
                    subtext="recursive improvement"
                    trend="up"
                />
                <MetricCard
                    label="Avg Iterations"
                    value={(analytics.avg_iterations || 0).toFixed(1)}
                    subtext="per query"
                />
                <MetricCard
                    label="Total Cost"
                    value={`$${(analytics.total_cost || 0).toFixed(2)}`}
                    subtext="estimated compute"
                />
            </div>

            {/* Main Chart Area */}
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '1.5rem', height: '400px', marginBottom: '2rem' }}>
                {/* Quality Trend Chart */}
                <GlassPanel intensity="medium" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
                    <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', color: 'var(--color-text-secondary)' }}>
                        Real-time Quality Velocity
                    </h3>
                    <div style={{ flex: 1 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={chartData}>
                                <defs>
                                    <linearGradient id="colorQuality" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="var(--color-accent-primary)" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="var(--color-accent-primary)" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                                <XAxis dataKey="time" stroke="var(--color-text-secondary)" fontSize={12} tickLine={false} axisLine={false} />
                                <YAxis stroke="var(--color-text-secondary)" fontSize={12} tickLine={false} axisLine={false} unit="%" />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: 'var(--glass-bg)',
                                        borderColor: 'var(--glass-border)',
                                        backdropFilter: 'blur(10px)',
                                        color: 'var(--color-text-primary)'
                                    }}
                                />
                                <Area
                                    type="monotone"
                                    dataKey="quality"
                                    stroke="var(--color-accent-primary)"
                                    strokeWidth={2}
                                    fillOpacity={1}
                                    fill="url(#colorQuality)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </GlassPanel>

                {/* Top Strategies List */}
                <GlassPanel intensity="medium" style={{ padding: '1.5rem' }}>
                    <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', color: 'var(--color-text-secondary)' }}>
                        Top Strategies
                    </h3>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        {analytics.strategies ? (
                            Object.entries(analytics.strategies)
                                .sort((a, b) => b[1].count - a[1].count)
                                .slice(0, 5)
                                .map(([name, stats]) => (
                                    <div key={name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <span style={{ fontSize: '0.9rem' }}>{name}</span>
                                        <div style={{ textAlign: 'right' }}>
                                            <div style={{ fontWeight: 'bold' }}>{stats.count}</div>
                                            <div style={{ fontSize: '0.8rem', color: 'var(--color-accent-primary)' }}>
                                                {((stats.avg_quality_gain || 0) * 100).toFixed(0)}% gain
                                            </div>
                                        </div>
                                    </div>
                                ))
                        ) : (
                            <div style={{ color: 'var(--color-text-muted)', fontSize: '0.9rem' }}>No strategy data yet</div>
                        )}
                    </div>
                </GlassPanel>
            </div>

            {/* Recent Activity Table */}
            <GlassPanel intensity="medium" style={{ padding: '1.5rem' }}>
                <h3 style={{ margin: '0 0 1rem 0', fontSize: '1rem', color: 'var(--color-text-secondary)' }}>
                    Recent Executions
                </h3>
                <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                        <thead>
                            <tr style={{ borderBottom: '1px solid var(--color-border)', textAlign: 'left' }}>
                                <th style={{ padding: '0.75rem', color: 'var(--color-text-secondary)' }}>Time</th>
                                <th style={{ padding: '0.75rem', color: 'var(--color-text-secondary)' }}>Strategy</th>
                                <th style={{ padding: '0.75rem', color: 'var(--color-text-secondary)' }}>Query</th>
                                <th style={{ padding: '0.75rem', color: 'var(--color-text-secondary)' }}>Iter</th>
                                <th style={{ padding: '0.75rem', color: 'var(--color-text-secondary)' }}>Gain</th>
                            </tr>
                        </thead>
                        <tbody>
                            {recentExecutions && recentExecutions.length > 0 ? (
                                recentExecutions.map((exec, i) => (
                                    <tr key={i} style={{ borderBottom: '1px solid var(--glass-border)' }}>
                                        <td style={{ padding: '0.75rem', color: 'var(--color-text-muted)' }}>
                                            {new Date(exec.timestamp).toLocaleTimeString()}
                                        </td>
                                        <td style={{ padding: '0.75rem' }}>{exec.strategy}</td>
                                        <td style={{ padding: '0.75rem' }}>
                                            {exec.query_text.substring(0, 40)}{exec.query_text.length > 40 ? '...' : ''}
                                        </td>
                                        <td style={{ padding: '0.75rem' }}>{exec.iterations}</td>
                                        <td style={{ padding: '0.75rem', color: (exec.quality_gain > 0 ? 'var(--color-accent-primary)' : 'inherit') }}>
                                            {((exec.quality_gain || 0) * 100).toFixed(1)}%
                                        </td>
                                    </tr>
                                ))
                            ) : (
                                <tr>
                                    <td colspan="5" style={{ padding: '2rem', textAlign: 'center', color: 'var(--color-text-muted)' }}>
                                        No recent activity
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </GlassPanel>
        </div>
    );
};

export default AnalyticsDashboard;
