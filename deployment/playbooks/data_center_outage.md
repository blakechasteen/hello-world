# Data Center Outage Recovery Playbook

**Created**: 2025-11-16
**Last Updated**: 2025-11-16
**RTO Target**: 5 minutes (automatic failover)
**RPO Target**: 0 minutes (real-time replication)

## Overview

This playbook covers recovery from complete data center or availability zone outages, requiring immediate failover to backup regions with zero/minimal data loss.

## Severity: CRITICAL

**Impact**:
- Complete regional failure
- All services in region unavailable
- Requires immediate multi-region failover
- Potential for split-brain if partial outage

## Prerequisites

- Multi-region deployment active
- Real-time replication configured
- Failover manager operational
- Access to backup region infrastructure
- Global load balancer with health-based routing

## Types of Data Center Outages

### 1. Availability Zone Failure
- **Scope**: Single AZ (e.g., us-east-1a)
- **Impact**: 33% capacity if 3-AZ deployment
- **Recovery**: Automatic (load balancer routes to healthy AZs)
- **RTO**: < 2 minutes

### 2. Complete Regional Outage
- **Scope**: Entire region (e.g., all of us-east-1)
- **Impact**: 100% primary capacity lost
- **Recovery**: Failover to secondary region (us-west-2)
- **RTO**: < 5 minutes

### 3. Cloud Provider Outage
- **Scope**: Multiple regions or all AWS
- **Impact**: All cloud infrastructure unavailable
- **Recovery**: Failover to different provider (GCP, Azure)
- **RTO**: 15-30 minutes

## Detection

### Automatic Detection

Failover manager automatically detects outages via:

```python
# HoloLoom/voice/failover.py monitors:
# - Health check failures (3 consecutive = region down)
# - Network connectivity
# - Response latency > threshold
# - Cloud provider status APIs
```

### Manual Detection Indicators

```bash
# AWS Service Health Dashboard
https://status.aws.amazon.com/

# Monitor shows:
# - [IMPACT] EC2: Increased error rates in us-east-1
# - [IMPACT] VPC: Connectivity issues in us-east-1a

# Your monitoring alerts:
# - Grafana: All us-east-1 metrics flatline
# - PagerDuty: Multiple alerts firing
# - Uptime checks: All external probes failing
```

## Automatic Failover

### How It Works

1. **Health Checks Fail** (2 minutes)
   ```
   Failover Manager detects:
   - us-east-1 health check timeout
   - 3 consecutive failures
   - Latency > 5000ms
   ```

2. **Automatic Failover Triggered** (30 seconds)
   ```python
   # Failover manager automatically:
   manager.active_region = regions[1]  # us-west-2
   # Callback triggers:
   on_failover(old_region="us-east-1", new_region="us-west-2")
   ```

3. **DNS/LB Update** (1 minute)
   ```
   Global Load Balancer:
   - Detects us-east-1 health check failures
   - Routes all traffic to us-west-2
   - TTL: 60 seconds
   ```

4. **Traffic Flows to Backup** (2 minutes)
   ```
   Users automatically routed to:
   voice.hololoom.ai → us-west-2
   ```

**Total Automatic RTO**: **5 minutes**

## Manual Failover (If Automatic Fails)

### Step 1: Confirm Outage

**Time**: 1-2 minutes

```bash
# Check primary region
curl https://voice-us-east.hololoom.ai/health
# Expected: Timeout or 5xx errors

# Check cloud provider status
curl https://status.aws.amazon.com/data.json | \
  jq '.current_events[] | select(.region == "us-east-1")'

# Check monitoring dashboards
open http://monitoring.hololoom.ai/regions
```

### Step 2: Activate Secondary Region

**Time**: 2 minutes

```bash
# SSH to failover control server
ssh ops@failover-control.hololoom.ai

# Activate us-west-2 failover
cd /opt/hololoom-failover
python3 activate_failover.py \
  --from us-east-1 \
  --to us-west-2 \
  --reason "Data center outage - AWS Service Health: EC2 issues"

# Script performs:
# 1. Updates failover manager state
# 2. Updates DNS records
# 3. Updates global load balancer
# 4. Verifies us-west-2 health
# 5. Sends notifications
```

### Step 3: Update Global DNS

**Time**: 1 minute (if not automated)

```bash
# Update Route53 to point to us-west-2
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "voice.hololoom.ai",
        "Type": "CNAME",
        "TTL": 60,
        "ResourceRecords": [{"Value": "voice-us-west.hololoom.ai"}]
      }
    }]
  }'

# Verify DNS propagation
dig voice.hololoom.ai +short
# Should show: voice-us-west.hololoom.ai → [us-west-2 IP]
```

### Step 4: Verify Secondary Region

**Time**: 2 minutes

```bash
# Test secondary region health
curl https://voice-us-west.hololoom.ai/health | jq '.'

# Should show:
{
  "status": "healthy",
  "region": "us-west-2",
  "role": "active",
  "components": {
    "voice_agent": "ok",
    "neo4j": "ok",
    "redis": "ok"
  }
}

# Test end-to-end functionality
curl -X POST https://voice-us-west.hololoom.ai/voice/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Failover test query",
    "language": "en"
  }' | jq '.'
```

### Step 5: Monitor Traffic Migration

**Time**: 5 minutes

```bash
# Watch DNS TTL expire and traffic migrate
watch -n 5 'curl -s https://voice.hololoom.ai/health | jq ".region"'

# Monitor Grafana
open https://grafana.hololoom.ai/d/regional-traffic

# Verify:
# - us-east-1 traffic → 0 req/s
# - us-west-2 traffic → [normal load] req/s
# - Error rate stable
# - Latency stable
```

## Data Synchronization

### Real-Time Replication

**Continuous replication** ensures RPO ≈ 0:

```yaml
# Neo4j Cluster
us-east-1 (primary)
  ↓ real-time replication
us-west-2 (secondary)

# Redis Cache
- Independent per region
- Rebuilds from queries (acceptable)

# Prometheus/Grafana
- Federated storage (Thanos)
- All metrics available in both regions
```

### Verify Data Consistency

```bash
# Check Neo4j replication lag
curl -u neo4j:password https://voice-us-west.hololoom.ai/neo4j/metrics | \
  jq '.replication_lag_seconds'
# Should be < 1 second

# Query both regions for same data
QUERY='MATCH (n:Entity {name: "test"}) RETURN n.updated_at'

# Primary (if still accessible)
cypher-shell -a neo4j://us-east-1 -u neo4j -p password "$QUERY"

# Secondary
cypher-shell -a neo4j://us-west-2 -u neo4j -p password "$QUERY"

# Compare timestamps (should be identical or < 1s apart)
```

## Post-Failover Operations

### Immediate (First Hour)

1. **Monitor Secondary Region**
   ```bash
   # Watch metrics closely
   - Request rate
   - Error rate
   - Latency p50/p95/p99
   - Resource utilization (CPU, memory, disk)
   ```

2. **Scale Secondary if Needed**
   ```bash
   # If handling 2x normal load, scale up
   kubectl scale deployment hololoom-voice-agent --replicas=6

   # Or via auto-scaling
   kubectl autoscale deployment hololoom-voice-agent \
     --min=3 --max=10 --cpu-percent=70
   ```

3. **Communication**
   ```
   Send status update:
   - To: engineering@hololoom.ai, customers@hololoom.ai
   - Subject: [UPDATE] Service Failover - System Operational
   - Body: See template below
   ```

### Short-Term (First 24 Hours)

1. **Monitor Primary Region Recovery**
   ```bash
   # Check AWS status page hourly
   curl https://status.aws.amazon.com/data.json | \
     jq '.current_events[] | select(.region == "us-east-1")'

   # When AWS reports "Resolved":
   # - Verify primary region health
   # - Run validation tests
   # - Wait 2 hours for stability
   ```

2. **Plan Failback** (when primary recovered)
   ```bash
   # See failback procedure below
   ```

### Long-Term (First Week)

1. **Post-Mortem**
   - Schedule within 3 days
   - Document timeline
   - Identify improvements

2. **Update Documentation**
   - Update playbooks with lessons learned
   - Improve automation
   - Add new monitoring

## Failback to Primary

**When**: Primary region fully recovered and stable for 2+ hours

### Step 1: Verify Primary Health

```bash
# Comprehensive health check
bash scripts/validate_deployment.sh \
  --region us-east-1 \
  --comprehensive

# Should show all green:
# ✓ Neo4j cluster healthy
# ✓ Redis operational
# ✓ Voice Agent responding
# ✓ Elle AR functional
# ✓ All health checks passing
```

### Step 2: Sync Data (if needed)

```bash
# Check replication lag
# Primary should catch up to secondary during outage
cypher-shell -a neo4j://us-east-1 \
  "CALL dbms.cluster.overview()"

# If lag > 1 hour, manual sync:
neo4j-admin database restore \
  --from=backup-from-us-west-2
```

### Step 3: Gradual Failback

```bash
# Don't failback all at once - use gradual shift

# Route 10% traffic to us-east-1
aws elbv2 modify-target-group-attribute \
  --target-group-arn arn:... \
  --attributes Key=stickiness.enabled,Value=true

# Monitor for 30 minutes

# If stable, increase to 50%
# Monitor for 30 minutes

# If stable, increase to 100%
python3 activate_failover.py \
  --from us-west-2 \
  --to us-east-1 \
  --gradual \
  --duration 30m
```

### Step 4: Update DNS Back to Primary

```bash
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "voice.hololoom.ai",
        "Type": "CNAME",
        "TTL": 60,
        "ResourceRecords": [{"Value": "voice-us-east.hololoom.ai"}]
      }
    }]
  }'
```

## Communication Templates

### Initial Outage Notification

```
Subject: [INCIDENT] HoloLoom Voice Service - Regional Failover in Progress

Team,

We are currently experiencing an outage in our primary region (us-east-1)
due to [AWS data center issues / network connectivity / etc.].

Actions Taken:
- Automatic failover to us-west-2 initiated at [TIME]
- All traffic being routed to backup region
- Expected service restoration: [TIME] ([DURATION] minutes)

Impact:
- Brief service interruption (2-5 minutes)
- No data loss expected (real-time replication)
- All functionality available in backup region

Current Status:
- us-west-2: OPERATIONAL
- us-east-1: DOWN (monitoring for recovery)

We will send updates every 30 minutes until resolved.

Updates: #incidents channel
Questions: ops@hololoom.ai

- HoloLoom Operations Team
```

### Recovery Complete Notification

```
Subject: [RESOLVED] HoloLoom Voice Service - Failover Complete

Team,

The regional outage has been resolved. Service is fully operational on
backup infrastructure.

Timeline:
- Outage Detected: [TIME]
- Failover Initiated: [TIME]
- Service Restored: [TIME]
- Total Downtime: [DURATION] minutes

Final Status:
- Active Region: us-west-2
- RTO Achieved: [ACTUAL] (Target: 5 minutes)
- RPO Achieved: 0 minutes (no data loss)
- All systems operational

Next Steps:
- Continue monitoring us-east-1 recovery
- Plan failback when primary region stable
- Post-mortem scheduled for [DATE]

Thank you for your patience.

- HoloLoom Operations Team
```

## Post-Mortem Template

```markdown
# Data Center Outage Post-Mortem

**Date**: 2025-XX-XX
**Duration**: [DURATION]
**Severity**: CRITICAL

## Summary
[Brief description of what happened]

## Timeline
- [TIME] - Outage detected in us-east-1
- [TIME] - Automatic failover initiated
- [TIME] - DNS updated to us-west-2
- [TIME] - Service fully restored
- [TIME] - Primary region recovered

## Impact
- Users affected: [NUMBER / PERCENTAGE]
- Downtime: [DURATION]
- Data loss: [NONE / DESCRIPTION]
- Revenue impact: $[AMOUNT]

## Root Cause
[Detailed description - AWS issues, network, etc.]

## What Went Well
- Automatic failover worked as designed
- Real-time replication prevented data loss
- RTO target met (5 minutes)
- Communication was timely

## What Can Improve
- [Improvement 1]
- [Improvement 2]
- [Improvement 3]

## Action Items
- [ ] [Action 1] - Owner: [NAME] - Due: [DATE]
- [ ] [Action 2] - Owner: [NAME] - Due: [DATE]
- [ ] [Action 3] - Owner: [NAME] - Due: [DATE]

## Lessons Learned
[Key takeaways]
```

## Monitoring & Alerts

### Critical Alerts

```yaml
# Prometheus alerts
- alert: DataCenterOutage
  expr: up{region="us-east-1"} == 0
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Data center outage detected"
    description: "us-east-1 completely down - failover required"

- alert: FailoverActivated
  expr: failover_active{from_region="us-east-1"} == 1
  labels:
    severity: critical
  annotations:
    summary: "Automatic failover activated"
    description: "Traffic routed from us-east-1 to us-west-2"
```

## Prevention & Resilience

### 1. Multi-Region Architecture

Always deploy to ≥2 regions:
```yaml
Regions:
  Primary: us-east-1 (3 AZs)
  Secondary: us-west-2 (3 AZs)
  Tertiary: eu-west-1 (optional)
```

### 2. Chaos Engineering

Regular disaster drills:
```bash
# Monthly: Simulate region failure
chaos-engineering/simulate_region_outage.sh --region us-east-1 --duration 10m
```

### 3. Real-Time Replication

Configure for RPO ≈ 0:
```yaml
Neo4j:
  replication: synchronous
  min_replicas: 2
  ack_timeout: 1s
```

### 4. Automated Failover

Ensure failover manager is always operational:
```yaml
Failover Manager:
  deployment: multi-region (runs in both regions)
  health_check_interval: 10s
  auto_failover: true
```

## References

- [AWS Disaster Recovery](https://aws.amazon.com/disaster-recovery/)
- [Multi-Region Deployment Guide](../README.md#multi-region)
- [Failover Manager Documentation](../../HoloLoom/voice/failover.py)
- [Netflix Chaos Engineering](https://netflix.github.io/chaosmonkey/)

---

**Last Reviewed**: 2025-11-16
**Reviewer**: Agent H - Wave 3 Production Hardening
**Next Review**: 2025-12-16
