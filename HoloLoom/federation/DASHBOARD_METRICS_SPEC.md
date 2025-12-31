# Federation Seed Network Dashboard Specification
## Real-Time Operational Metrics & Visualization

**Status**: Implementation-ready
**Primary Tool**: Grafana (public read-only) + custom HTML dashboard
**Update Interval**: 60 seconds (near real-time)
**Audience**: Network operators, safety researchers, public observers

---

## Dashboard Architecture

```
┌─────────────────────────────────────────────────────┐
│         Metrics Data Collection (Each Node)          │
│ - Node exporter (system metrics)                     │
│ - Federation exporter (consensus, guild metrics)     │
│ - Audit trail collector (safety events)              │
└────────────┬────────────────────────────────────────┘
             │ Prometheus scrape (pull every 60s)
             ↓
┌─────────────────────────────────────────────────────┐
│       Prometheus Time-Series Database                │
│ - prometheus.federation.hololoom.dev                 │
│ - 30-day retention (configurable)                    │
│ - 5-minute aggregates for long-term                  │
└────────────┬────────────────────────────────────────┘
             │
    ┌────────┼────────┐
    │        │        │
    ↓        ↓        ↓
 ┌──────┐ ┌─────┐ ┌────────┐
 │Grafana│ │HTML │ │ CLI    │
 │Dash   │ │Dash │ │Tool    │
 │(Web)  │ │(Web)│ │(Term)  │
 └──────┘ └─────┘ └────────┘
```

---

## Part 1: Core Metrics Definitions

### 1.1 Network Health Metrics

#### SWIM Gossip Protocol Health

```yaml
federation_node_status:
  description: "Current status of each federation node"
  type: gauge
  labels:
    - node_id: "seed-bootstrap-us-w1" | "seed-eu-de1" | "seed-apac-sg1"
    - region: "us-w1" | "eu-de1" | "ap-sg1"
  values: [ALIVE=1, SUSPECT=2, DEAD=0]

  example:
    federation_node_status{node_id="seed-bootstrap-us-w1"} = 1
    federation_node_status{node_id="seed-eu-de1"} = 1
    federation_node_status{node_id="seed-apac-sg1"} = 1

federation_peers_discovered:
  description: "Number of peers each node knows about"
  type: gauge
  labels:
    - node_id
  values: [0, 1, 2, 3]  # Target: all nodes see all other nodes

  example:
    federation_peers_discovered{node_id="seed-bootstrap-us-w1"} = 3

federation_failure_detection_time_ms:
  description: "Time to detect node failure (latency of SWIM protocol)"
  type: histogram
  quantiles: [p50, p95, p99]
  target: p95 < 1000ms  # Should detect within 1 second

  example:
    federation_failure_detection_time_ms{quantile="0.50"} = 423
    federation_failure_detection_time_ms{quantile="0.95"} = 834
    federation_failure_detection_time_ms{quantile="0.99"} = 1342

federation_message_loss_rate:
  description: "Fraction of gossip messages lost"
  type: gauge
  labels:
    - source_node
    - dest_node
  target: <0.5% (high-quality network)

  example:
    federation_message_loss_rate{source="us-w1", dest="eu-de1"} = 0.003
    federation_message_loss_rate{source="eu-de1", dest="ap-sg1"} = 0.002

federation_gossip_rounds_total:
  description: "Cumulative SWIM gossip cycles"
  type: counter
  labels:
    - node_id

  example:
    federation_gossip_rounds_total{node_id="seed-bootstrap-us-w1"} = 47293
```

**Alert Rules**:
```promql
# Alert if node status is DEAD for >1 minute
alert: FederationNodeDown
expr: federation_node_status == 0
for: 1m
severity: critical

# Alert if failure detection is slow (>2 seconds)
alert: FailureDetectionSlow
expr: federation_failure_detection_time_ms{quantile="0.95"} > 2000
for: 5m
severity: warning

# Alert if message loss >1%
alert: HighMessageLoss
expr: federation_message_loss_rate > 0.01
for: 5m
severity: warning
```

### 1.2 Consensus & Verification Metrics

```yaml
federation_verification_latency_ms:
  description: "Time to reach consensus on response safety"
  type: histogram
  labels:
    - response_type: "factual" | "procedural" | "policy" | "other"
  quantiles: [p50, p95, p99, p99.9]
  target: p95 < 500ms

  example:
    # FAST verifications complete in 200-400ms
    federation_verification_latency_ms{response_type="factual", quantile="0.50"} = 287
    federation_verification_latency_ms{response_type="factual", quantile="0.95"} = 412
    federation_verification_latency_ms{response_type="factual", quantile="0.99"} = 589

    # Complex verifications take longer
    federation_verification_latency_ms{response_type="policy", quantile="0.95"} = 823

federation_verifications_total:
  description: "Cumulative verifications by status"
  type: counter
  labels:
    - node_id
    - status: "success" | "failed" | "timeout" | "cancelled"

  example:
    federation_verifications_total{node_id="seed-bootstrap-us-w1", status="success"} = 1244
    federation_verifications_total{node_id="seed-bootstrap-us-w1", status="failed"} = 3
    federation_verifications_total{node_id="seed-bootstrap-us-w1", status="timeout"} = 0

federation_consensus_reached:
  description: "Consensus was reached (verification succeeded)"
  type: counter
  labels:
    - reason: "unanimous" | "strong_agreement" | "majority"

  example:
    # UNANIMOUS: All 3 verifiers agreed
    federation_consensus_reached{reason="unanimous"} = 1087
    # STRONG: 3 of 4 verifiers agreed (we have 3 nodes)
    # (Not used with 3 nodes, but shown for future expansion)
    federation_consensus_reached{reason="strong_agreement"} = 157
    # MAJORITY: 2 of 3 verifiers agreed
    federation_consensus_reached{reason="majority"} = 3

federation_consensus_failed:
  description: "Consensus failed (verification blocked)"
  type: counter
  labels:
    - reason: "timeout" | "network_partition" | "verification_failed"

  example:
    federation_consensus_failed{reason="timeout"} = 0
    federation_consensus_failed{reason="verification_failed"} = 0

federation_safety_checks_total:
  description: "Safety validation passed/failed"
  type: counter
  labels:
    - node_id
    - result: "passed" | "failed"

  example:
    federation_safety_checks_total{node_id="seed-bootstrap-us-w1", result="passed"} = 1244
    federation_safety_checks_total{node_id="seed-bootstrap-us-w1", result="failed"} = 0
```

**Derived Metrics** (calculated in Grafana):
```
Success Rate = sum(federation_verifications_total{status="success"})
             / sum(federation_verifications_total)

Agreement Level (avg) = sum(federation_consensus_reached{reason="unanimous"})
                      / sum(federation_consensus_reached)
```

### 1.3 Guild & Reputation Metrics

```yaml
guild_members_total:
  description: "Number of guild members by status"
  type: gauge
  labels:
    - guild_id: "safety-researchers-founding-001"
    - status: "active" | "suspended" | "inactive"

  example:
    guild_members_total{guild_id="safety-researchers-founding-001", status="active"} = 15
    guild_members_total{guild_id="safety-researchers-founding-001", status="suspended"} = 0
    guild_members_total{guild_id="safety-researchers-founding-001", status="inactive"} = 2

node_reputation:
  description: "Current reputation score for each node (0.0-1.0)"
  type: gauge
  labels:
    - node_id

  example:
    node_reputation{node_id="seed-bootstrap-us-w1"} = 0.87
    node_reputation{node_id="seed-eu-de1"} = 0.82
    node_reputation{node_id="seed-apac-sg1"} = 0.91

node_reputation_change_total:
  description: "Cumulative reputation gains/losses"
  type: counter
  labels:
    - node_id
    - direction: "up" | "down"

  example:
    node_reputation_change_total{node_id="seed-bootstrap-us-w1", direction="up"} = 0.37
    node_reputation_change_total{node_id="seed-bootstrap-us-w1", direction="down"} = -0.04

node_verifications_attempted:
  description: "Total verification attempts (success + failed)"
  type: counter
  labels:
    - node_id

  example:
    node_verifications_attempted{node_id="seed-bootstrap-us-w1"} = 1247

node_days_active:
  description: "Days since node joined founding guild"
  type: gauge
  labels:
    - node_id

  example:
    node_days_active{node_id="seed-bootstrap-us-w1"} = 30
```

**Derived Metrics**:
```
Verification Success Rate (per node) =
  sum(federation_verifications_total{node_id="X", status="success"})
  / sum(federation_verifications_total{node_id="X"})

Average Reputation = avg(node_reputation)

Reputation Std Dev = stddev(node_reputation)
```

### 1.4 Operational Health Metrics

```yaml
federation_node_uptime_percent:
  description: "Monthly rolling uptime percentage"
  type: gauge
  labels:
    - node_id
  target: >99.9% (2.7 hours/month allowed downtime)

  example:
    federation_node_uptime_percent{node_id="seed-bootstrap-us-w1"} = 99.95
    federation_node_uptime_percent{node_id="seed-eu-de1"} = 99.92
    federation_node_uptime_percent{node_id="seed-apac-sg1"} = 99.98

federation_response_generation_time_ms:
  description: "Time for node to generate response (excluding network)"
  type: histogram
  labels:
    - node_id
  quantiles: [p50, p95, p99]

  example:
    federation_response_generation_time_ms{node_id="seed-bootstrap-us-w1", quantile="0.50"} = 87
    federation_response_generation_time_ms{node_id="seed-bootstrap-us-w1", quantile="0.95"} = 156

federation_network_bandwidth_mbps:
  description: "Network throughput to/from peers"
  type: gauge
  labels:
    - node_id
    - direction: "in" | "out"

  example:
    federation_network_bandwidth_mbps{node_id="seed-bootstrap-us-w1", direction="in"} = 2.3
    federation_network_bandwidth_mbps{node_id="seed-bootstrap-us-w1", direction="out"} = 2.1

node_disk_usage_percent:
  description: "Disk space used (alert if >80%)"
  type: gauge
  labels:
    - node_id

  example:
    node_disk_usage_percent{node_id="seed-bootstrap-us-w1"} = 34.2

node_memory_usage_percent:
  description: "RAM usage (alert if >75%)"
  type: gauge
  labels:
    - node_id

  example:
    node_memory_usage_percent{node_id="seed-bootstrap-us-w1"} = 62.1
```

---

## Part 2: Grafana Dashboard Design

### Dashboard 1: Network Overview

**Layout**: 4 rows × 2 columns

```
┌─────────────────────────────────────────────────────────────┐
│ HoloLoom Federation Dashboard - Network Overview             │
│ Last Updated: 2026-01-15 14:23:45 UTC  |  Refresh: 60s     │
├─────────────────────────────────────────────────────────────┤
│ ROW 1: Network Status                                       │
│ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐   │
│ │ Node #1        │ │ Node #2        │ │ Node #3        │   │
│ │ 🟢 ALIVE       │ │ 🟢 ALIVE       │ │ 🟢 ALIVE       │   │
│ │ 168.90.25.45   │ │ 151.39.20.117  │ │ 45.142.89.201  │   │
│ │ Peers: 2       │ │ Peers: 2       │ │ Peers: 2       │   │
│ │ Uptime: 99.95% │ │ Uptime: 99.92% │ │ Uptime: 99.98% │   │
│ └────────────────┘ └────────────────┘ └────────────────┘   │
│                                                             │
│ ROW 2: Failure Detection & Messaging                       │
│ ┌────────────────────────┐ ┌────────────────────────────┐  │
│ │ Failure Detection      │ │ Message Loss Rate          │  │
│ │ Time (p95):            │ │ (24h average):             │  │
│ │ 834 ms ▲ (good)        │ │ 0.3% ▼ (excellent)         │  │
│ │ Target: <1000ms        │ │ Target: <0.5%              │  │
│ └────────────────────────┘ └────────────────────────────┘  │
│                                                             │
│ ROW 3: Peer Discovery                                      │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Peers Discovered                                     │   │
│ │ us-w1: 3/3 ✓  eu-de1: 3/3 ✓  ap-sg1: 3/3 ✓        │   │
│ │ All nodes aware of all peers (full mesh)             │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ROW 4: Gossip Activity                                     │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Gossip Rounds (Cumulative)                           │   │
│ │ 50000 ┤       ╭─────────                             │   │
│ │       │     ╭─╯                                      │   │
│ │ 45000 ├───╭─╯                                        │   │
│ │       │ ╭─╯                                          │   │
│ │ 40000 ├─╯                                            │   │
│ │       │ us-w1 (47293)                                │   │
│ │       │ eu-de1 (47401)                               │   │
│ │       │ ap-sg1 (47156)                               │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Dashboard 2: Consensus Performance

```
┌─────────────────────────────────────────────────────────────┐
│ Consensus Performance & Verification Latency                │
├─────────────────────────────────────────────────────────────┤
│ ROW 1: Verification Metrics (24h)                          │
│ ┌────────────────────────┐ ┌────────────────────────────┐  │
│ │ Verifications Completed│ │ Success Rate               │  │
│ │ 1,247 (↑ +87 from 6h)  │ │ 99.8% (1,244/1,247) ✓    │  │
│ │ Rate: 52/hour          │ │ Failed: 3                  │  │
│ │                        │ │ Timeouts: 0                │  │
│ └────────────────────────┘ └────────────────────────────┘  │
│                                                             │
│ ROW 2: Latency Percentiles                                 │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Verification Latency (by type)                       │   │
│ │                                                       │   │
│ │ factual:    p50: 287ms  p95: 412ms  p99: 589ms  ✓    │   │
│ │ procedural: p50: 345ms  p95: 521ms  p99: 823ms      │   │
│ │ policy:     p50: 412ms  p95: 823ms  p99: 1342ms     │   │
│ │ other:      p50: 312ms  p95: 467ms  p99: 734ms      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ROW 3: Consensus Reasons (last 1000)                       │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Consensus Breakdown                                  │   │
│ │ ┌─────────────────────────────┐                      │   │
│ │ │ ■ UNANIMOUS (87.2%): 1,087   │                      │   │
│ │ │ ■ STRONG_AGR (12.6%): 157   │ ← 3-of-4 agreement  │   │
│ │ │ ■ MAJORITY (0.2%): 3        │ ← 2-of-3 agreement  │   │
│ │ └─────────────────────────────┘                      │   │
│ │ All 3 seed nodes necessary for consensus at launch   │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ROW 4: Safety Checks Passed/Failed                         │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 1400 ┤       ╭──────────                             │   │
│ │      │     ╭─╯                                       │   │
│ │ 1200 ├───╭─╯                                         │   │
│ │      │ ╭─╯                                           │   │
│ │ 1000 ├─╯                                             │   │
│ │      │ Passed:  1,244 (99.76%) ✓                    │   │
│ │      │ Failed:  0                                    │   │
│ │      │ None:    3 (0.24%) - network issues          │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Dashboard 3: Guild & Reputation

```
┌─────────────────────────────────────────────────────────────┐
│ Guild Status & Reputation Tracking                          │
├─────────────────────────────────────────────────────────────┤
│ ROW 1: Guild Metrics                                        │
│ ┌────────────────────────┐ ┌────────────────────────────┐  │
│ │ Guild Members          │ │ Member Distribution        │  │
│ │ Total: 15              │ │                            │  │
│ │ ├─ Active: 15          │ │ Veteran (0.80-0.98):  8   │  │
│ │ ├─ Suspended: 0        │ │ Established (0.60-0.80): 5│  │
│ │ └─ Inactive: 2         │ │ Starter (0.40-0.60): 2    │  │
│ │                        │ │ Probation (<0.40): 0       │  │
│ └────────────────────────┘ └────────────────────────────┘  │
│                                                             │
│ ROW 2: Reputation Statistics                               │
│ ┌────────────────────────┐ ┌────────────────────────────┐  │
│ │ Reputation Distribution│ │ Individual Node Reputations│  │
│ │                        │ │                            │  │
│ │ μ (mean): 0.786        │ │ seed-bootstrap-us-w1: 0.87 │  │
│ │ σ (std):  0.081        │ │ seed-eu-de1:         0.82  │  │
│ │ min:      0.45         │ │ seed-apac-sg1:       0.91  │  │
│ │ max:      0.94         │ │                            │  │
│ │ Target σ: <0.15        │ │ (3 founding members)       │  │
│ └────────────────────────┘ └────────────────────────────┘  │
│                                                             │
│ ROW 3: Reputation Tracking (30-day)                        │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Reputation Change (all members)                      │   │
│ │ 0.95 ┤      ╭────────                                │   │
│ │      │    ╭─╯                                        │   │
│ │ 0.85 ├──╭─╯                                          │   │
│ │      │╭─╯                                            │   │
│ │ 0.75 ├╯                                              │   │
│ │      │ (3 safety incidents last 30 days)            │   │
│ │ 0.65 ├─────────────────────────────────             │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ROW 4: Pending Admissions                                  │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Applicants Pending                                   │   │
│ │ 1. [Org A] - Day 7/14 of onboarding                  │   │
│ │    └─ Sponsors: [Name1], [Name2]                     │   │
│ │ 2. [Org B] - Awaiting interview (scheduled Wed)      │   │
│ │    └─ Sponsors: [Name3], [Name4]                     │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Dashboard 4: Operational Health

```
┌─────────────────────────────────────────────────────────────┐
│ Node Operational Health & System Resources                  │
├─────────────────────────────────────────────────────────────┤
│ ROW 1: Node Uptime (rolling 30 days)                        │
│ ┌────────────────────────┐ ┌────────────────────────────┐  │
│ │ seed-bootstrap-us-w1   │ │ seed-eu-de1                │  │
│ │ 99.95% ✓ (SLA: 99.9%)  │ │ 99.92% ✓ (SLA: 99.9%)    │  │
│ │ Downtime: 34.5 min     │ │ Downtime: 43.2 min        │  │
│ └────────────────────────┘ └────────────────────────────┘  │
│                                                             │
│ ┌────────────────────────┐                                 │
│ │ seed-apac-sg1          │                                 │
│ │ 99.98% ✓ (SLA: 99.9%)  │                                 │
│ │ Downtime: 17.3 min     │                                 │
│ └────────────────────────┘                                 │
│                                                             │
│ ROW 2: System Resources                                    │
│ ┌────────────────────────┐ ┌────────────────────────────┐  │
│ │ Disk Usage             │ │ Memory Usage               │  │
│ │                        │ │                            │  │
│ │ us-w1:  34.2% ✓       │ │ us-w1:  62.1% ✓           │  │
│ │ eu-de1: 41.7% ✓       │ │ eu-de1: 58.3% ✓           │  │
│ │ ap-sg1: 29.1% ✓       │ │ ap-sg1: 55.2% ✓           │  │
│ │                        │ │                            │  │
│ │ (Alert: >80%)          │ │ (Alert: >75%)              │  │
│ └────────────────────────┘ └────────────────────────────┘  │
│                                                             │
│ ROW 3: Network Bandwidth (24h)                             │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Bandwidth Usage (IN / OUT)                           │   │
│ │ 3.0 Mbps ┤        ╭────                              │   │
│ │          │      ╭─╯                                  │   │
│ │ 2.0 Mbps ├────╭─╯                                    │   │
│ │          │  ╭─╯                                      │   │
│ │ 1.0 Mbps ├╭─╯                                        │   │
│ │          │                                           │   │
│ │ us-w1:   IN: 2.3 Mbps  OUT: 2.1 Mbps                │   │
│ │ eu-de1:  IN: 1.8 Mbps  OUT: 1.9 Mbps                │   │
│ │ ap-sg1:  IN: 1.2 Mbps  OUT: 1.1 Mbps                │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                             │
│ ROW 4: Response Generation Time                            │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Generation Time (p50 / p95)                          │   │
│ │                                                       │   │
│ │ us-w1:  p50: 87ms   p95: 156ms  ✓                   │   │
│ │ eu-de1: p50: 94ms   p95: 163ms  ✓                   │   │
│ │ ap-sg1: p50: 82ms   p95: 142ms  ✓                   │   │
│ │                                                       │   │
│ │ (Excludes network latency, pure generation time)    │   │
│ └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 3: HTML Custom Dashboard

**Endpoint**: https://federation.hololoom.dev/dashboard
**Technology**: Vanilla JS + Chart.js (no external deps)
**Refresh**: WebSocket (real-time, sub-second)
**Accessibility**: WCAG 2.1 AA

### HTML Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HoloLoom Federation Dashboard</title>
    <style>
        /* Tufte-style minimal design */
        body {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            background: #fefcf9;
            color: #333;
            margin: 2rem;
            line-height: 1.6;
        }

        .header {
            border-bottom: 2px solid #000;
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }

        .dashboard-section {
            margin: 3rem 0;
            padding: 1.5rem;
            background: #fff;
            border-left: 3px solid #e8e8e8;
        }

        .metric-card {
            display: inline-block;
            width: 30%;
            margin-right: 3%;
            vertical-align: top;
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #000;
        }

        .metric-label {
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
        }

        .status-alive { color: #2ecc71; }  /* Green */
        .status-suspect { color: #f39c12; }  /* Orange */
        .status-dead { color: #e74c3c; }  /* Red */
    </style>
</head>
<body>
    <div class="header">
        <h1>🌐 HoloLoom Federation Dashboard</h1>
        <p>Real-time operational metrics for decentralized safety verification network</p>
        <p style="color: #999; font-size: 0.9rem;">
            Last Updated: <span id="last-update">--:--:-- UTC</span> |
            Next Update: <span id="next-update">--</span>s
        </p>
    </div>

    <!-- Network Overview Section -->
    <section class="dashboard-section">
        <h2>Network Status</h2>

        <div class="metric-card">
            <div class="metric-label">Nodes Online</div>
            <div class="metric-value" id="nodes-online">-/-</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Failure Detection (p95)</div>
            <div class="metric-value" id="detection-latency">-- ms</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Message Loss Rate</div>
            <div class="metric-value" id="message-loss">-- %</div>
        </div>

        <div style="clear: both; margin-top: 2rem;">
            <h3>Node Status</h3>
            <div id="node-list">
                <!-- Populated by JS -->
            </div>
        </div>
    </section>

    <!-- Consensus Section -->
    <section class="dashboard-section">
        <h2>Consensus Performance (24h)</h2>

        <div class="metric-card">
            <div class="metric-label">Verifications</div>
            <div class="metric-value" id="verifications-total">--</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value" id="success-rate">-- %</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Latency (p95)</div>
            <div class="metric-value" id="consensus-latency">-- ms</div>
        </div>

        <div style="clear: both; margin-top: 2rem;">
            <canvas id="verification-chart" height="100"></canvas>
        </div>
    </section>

    <!-- Guild Section -->
    <section class="dashboard-section">
        <h2>Guild Status</h2>

        <div class="metric-card">
            <div class="metric-label">Members</div>
            <div class="metric-value" id="guild-members">--</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Avg Reputation</div>
            <div class="metric-value" id="avg-reputation">-- </div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Pending Applications</div>
            <div class="metric-value" id="pending-admission">--</div>
        </div>
    </section>

    <!-- Safety Audit Section -->
    <section class="dashboard-section">
        <h2>Safety Audit (7-day)</h2>

        <div class="metric-card">
            <div class="metric-label">Incidents</div>
            <div class="metric-value" id="safety-incidents">--</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Verification Failures</div>
            <div class="metric-value" id="verification-failures">--</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Consensus Timeouts</div>
            <div class="metric-value" id="consensus-timeouts">--</div>
        </div>

        <div style="clear: both; margin-top: 2rem;">
            <h3>Recent Events</h3>
            <div id="events-list" style="font-size: 0.9rem; font-family: monospace;">
                <!-- Populated by JS -->
            </div>
        </div>
    </section>

    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <script src="/js/dashboard.js"></script>
</body>
</html>
```

---

## Part 4: CLI Tool

**Command**: `federation status --metrics`

```bash
#!/bin/bash
# federation-metrics.sh - CLI tool for operators

set -e

PROMETHEUS_URL="http://localhost:9090"
REFRESH_RATE=5  # seconds

function format_metric() {
    local label=$1
    local value=$2
    local unit=$3
    printf "%-30s %15s %s\n" "$label:" "$value" "$unit"
}

function fetch_metric() {
    local query=$1
    curl -s "$PROMETHEUS_URL/api/v1/query?query=$query" | jq '.data.result[0].value[1]' -r
}

function show_dashboard() {
    clear

    echo "╭─ HoloLoom Federation Dashboard ─────────────────────────────────╮"
    echo "│ Last Updated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')                   │"
    echo "├──────────────────────────────────────────────────────────────────┤"

    # Network Status
    echo "│ NETWORK STATUS                                                   │"
    nodes_alive=$(fetch_metric 'sum(federation_node_status)')
    detection_ms=$(fetch_metric 'federation_failure_detection_time_ms{quantile="0.95"}')
    msg_loss=$(fetch_metric 'avg(federation_message_loss_rate)*100')

    echo "│  Nodes Alive: $nodes_alive/3                                           │"
    echo "│  Failure Detection (p95): ${detection_ms}ms                            │"
    echo "│  Message Loss Rate: ${msg_loss}%                                      │"

    echo "│                                                                  │"
    echo "│ CONSENSUS PERFORMANCE (24h)                                      │"
    verif_total=$(fetch_metric 'sum(federation_verifications_total)')
    success_rate=$(fetch_metric 'sum(federation_verifications_total{status="success"})/sum(federation_verifications_total)*100')
    latency=$(fetch_metric 'federation_verification_latency_ms{quantile="0.95"}')

    echo "│  Verifications: $verif_total                                             │"
    echo "│  Success Rate: ${success_rate}%                                       │"
    echo "│  Latency (p95): ${latency}ms                                           │"

    echo "│                                                                  │"
    echo "│ GUILD STATUS                                                     │"
    members=$(fetch_metric 'sum(guild_members_total{status="active"})')
    avg_rep=$(fetch_metric 'avg(node_reputation)')

    echo "│  Members: $members                                                      │"
    echo "│  Avg Reputation: ${avg_rep}                                            │"

    echo "│                                                                  │"
    echo "│ SAFETY AUDIT (7-day)                                             │"
    incidents=$(fetch_metric 'sum(federation_safety_incidents_total)')

    echo "│  Incidents: $incidents                                                  │"

    echo "╰──────────────────────────────────────────────────────────────────╯"
}

# Main loop
while true; do
    show_dashboard
    sleep $REFRESH_RATE
done
```

---

## Appendix A: Alert Rules (Prometheus)

```yaml
# /etc/prometheus/federation-alerts.yml

groups:
  - name: federation_critical
    rules:
      - alert: FederationNodeDown
        expr: federation_node_status == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Federation node {{ $labels.node_id }} is DOWN"
          description: "Node has been unavailable for >1 minute"

      - alert: FailureDetectionSlow
        expr: federation_failure_detection_time_ms{quantile="0.95"} > 2000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Failure detection is slow"
          description: "p95 detection time: {{ $value }}ms (target: <1000ms)"

      - alert: HighMessageLoss
        expr: avg(federation_message_loss_rate) > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High message loss detected"
          description: "Message loss rate: {{ $value }}% (target: <0.5%)"

      - alert: ConsensusFailures
        expr: increase(federation_consensus_failed[1h]) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Multiple consensus failures"
          description: "{{ $value }} failures in last hour"
```

---

## Conclusion

This dashboard specification provides:
- ✅ Production-grade metrics (20+ key indicators)
- ✅ Multi-format output (Grafana + HTML + CLI)
- ✅ Real-time updates (60-second sync)
- ✅ Operator-friendly interface (no learning curve)
- ✅ Complete transparency (all metrics public)

**Implementation order**:
1. Week 1: Deploy Prometheus + Node exporter
2. Week 2: Implement federation-specific metrics (in code)
3. Week 3: Configure Grafana dashboards (import JSON)
4. Week 4: Build custom HTML dashboard
5. Week 5: Add CLI tool

