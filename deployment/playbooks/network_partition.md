# Network Partition Recovery Playbook

**Created**: 2025-11-16
**Last Updated**: 2025-11-16
**RTO Target**: 15 minutes (automatic failover: 2 minutes)
**RPO Target**: 0 (no data loss)

## Overview

This playbook covers recovery from network partitions (split-brain scenarios) where components can't communicate due to network failures, but individual systems remain operational.

## Symptoms

- Intermittent connection failures
- Timeout errors between services
- Neo4j cluster split-brain warnings
- Redis connection refused
- Voice Agent can't reach Neo4j/Redis
- External clients can't reach services
- DNS resolution failures
- High packet loss or latency

## Types of Network Partitions

### 1. External Network Partition
- Clients can't reach HoloLoom → **DNS/Load balancer issue**
- Recovery: Update DNS, failover to backup region

### 2. Internal Network Partition
- Voice Agent ↔ Neo4j disconnected → **Service mesh issue**
- Recovery: Restart affected containers, check Docker network

### 3. Database Cluster Partition
- Neo4j cluster split-brain → **Cluster reconfiguration**
- Recovery: Force leader election, rebuild cluster

### 4. Multi-Region Partition
- us-east-1 ↔ us-west-2 disconnected → **Regional failover**
- Recovery: Activate failover manager

## Diagnosis

### Step 1: Identify Partition Type

**Time**: 2-3 minutes

```bash
# Test external connectivity
curl https://voice.hololoom.ai/health
# If fails → External partition

# Test from inside network
docker exec hololoom-voice-agent curl http://localhost:8000/health
# If succeeds → External partition confirmed

# Test internal connectivity
docker exec hololoom-voice-agent curl http://hololoom-neo4j:7474
docker exec hololoom-voice-agent curl http://hololoom-tts-cache:6379
# If fails → Internal partition

# Check Docker network
docker network inspect hololoom-network
# Verify all containers attached

# Test DNS resolution
docker exec hololoom-voice-agent nslookup hololoom-neo4j
docker exec hololoom-voice-agent ping -c 3 hololoom-neo4j
```

### Step 2: Check Network Layer

**Time**: 2 minutes

```bash
# Check host network interfaces
ip addr show
ifconfig

# Check routing table
ip route
route -n

# Test inter-container connectivity
docker exec hololoom-voice-agent ping -c 3 hololoom-neo4j
docker exec hololoom-voice-agent telnet hololoom-neo4j 7687

# Check firewall rules
iptables -L -n
ufw status

# Check for network errors
dmesg | grep -i network
journalctl -u docker | grep -i network
```

## Recovery Procedures

### Scenario A: External Network Partition

**RTO**: 2-5 minutes (automatic with failover manager)

#### Symptoms:
- External health checks fail
- Users can't reach service
- Internal connectivity OK

#### Recovery:

```bash
# Step 1: Verify internal health
docker exec hololoom-voice-agent curl http://localhost:8000/health
# Should return 200 OK

# Step 2: Check DNS
nslookup voice.hololoom.ai
# Should resolve to correct IP

# Step 3: Check load balancer
aws elbv2 describe-target-health \
  --target-group-arn arn:aws:elasticloadbalancing:...

# If unhealthy, re-register target
aws elbv2 register-targets \
  --target-group-arn arn:aws:elasticloadbalancing:... \
  --targets Id=$(ec2-metadata --instance-id | cut -d' ' -f2)

# Step 4: Activate failover manager (automatic)
# Failover manager detects partition and routes to healthy region
# See failover manager status:
curl http://localhost:8000/internal/failover/status | jq '.'
```

### Scenario B: Internal Service Partition

**RTO**: 5-10 minutes

#### Symptoms:
- Voice Agent can't reach Neo4j/Redis
- Connection refused errors in logs
- Docker network issues

#### Recovery:

```bash
# Step 1: Recreate Docker network
docker-compose -f docker-compose.voice.yml down
docker network prune -f
docker network create hololoom-network
docker-compose -f docker-compose.voice.yml up -d

# Step 2: Verify connectivity
docker exec hololoom-voice-agent ping -c 3 hololoom-neo4j
docker exec hololoom-voice-agent ping -c 3 hololoom-tts-cache

# Step 3: Restart affected services (if needed)
docker-compose -f docker-compose.voice.yml restart neo4j redis voice-agent

# Step 4: Verify health
curl http://localhost:8000/health | jq '.'
```

### Scenario C: Database Cluster Split-Brain

**RTO**: 10-15 minutes

#### Symptoms:
- Neo4j cluster has multiple leaders
- Inconsistent data across nodes
- Cluster health degraded

#### Recovery:

```bash
# Step 1: Identify cluster status
docker exec hololoom-neo4j cypher-shell \
  "CALL dbms.cluster.overview()"

# Step 2: Stop all secondary nodes
docker stop neo4j-secondary-1 neo4j-secondary-2

# Step 3: Force primary to be leader
docker exec hololoom-neo4j cypher-shell \
  "CALL dbms.cluster.setRole('LEADER')"

# Step 4: Restart secondary nodes one by one
docker start neo4j-secondary-1
sleep 30
docker start neo4j-secondary-2

# Step 5: Verify cluster healthy
docker exec hololoom-neo4j cypher-shell \
  "CALL dbms.cluster.overview()"
# All nodes should show FOLLOWER except one LEADER
```

### Scenario D: Multi-Region Partition

**RTO**: 2 minutes (automatic failover)

#### Symptoms:
- Cross-region connectivity lost
- Replication lag increasing
- Health checks failing in one region

#### Recovery:

```bash
# Failover manager automatically detects and fails over
# Manual override if needed:

python3 <<EOF
from HoloLoom.voice.failover import FailoverManager, Region

regions = [
    Region("us-east-1", "https://voice-us-east.hololoom.ai", priority=1),
    Region("us-west-2", "https://voice-us-west.hololoom.ai", priority=2),
]

manager = FailoverManager(regions)
await manager.start()

# Force failover to us-west-2
manager.active_region = regions[1]
print(f"Failed over to: {manager.active_region.name}")
EOF

# Update global load balancer
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
```

## Verification Steps

### Step 1: Test End-to-End Connectivity

**Time**: 3 minutes

```bash
# External → Voice Agent
curl https://voice.hololoom.ai/health

# Voice Agent → Neo4j
docker exec hololoom-voice-agent curl http://hololoom-neo4j:7474

# Voice Agent → Redis
docker exec hololoom-voice-agent redis-cli -h hololoom-tts-cache PING

# Test full query flow
curl -X POST https://voice.hololoom.ai/voice/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Network partition recovery test",
    "language": "en"
  }' | jq '.'
```

### Step 2: Monitor Network Metrics

```bash
# Packet loss
ping -c 100 voice.hololoom.ai | grep packet

# Latency
time curl https://voice.hololoom.ai/health

# Check Prometheus network metrics
curl http://localhost:9090/api/v1/query?query=network_errors_total
```

## Post-Recovery Checklist

- [ ] All external health checks passing
- [ ] All internal service connectivity restored
- [ ] No split-brain in database cluster
- [ ] Packet loss < 1%
- [ ] Latency < 100ms
- [ ] Failover manager shows all regions healthy
- [ ] No network errors in logs
- [ ] Metrics being collected normally
- [ ] End-to-end queries working

## Network Monitoring

### Key Metrics to Monitor

```yaml
# Prometheus alerts
- alert: NetworkPartitionDetected
  expr: up{job="voice-agent"} == 0
  for: 1m
  annotations:
    summary: "Service unreachable - possible network partition"

- alert: HighPacketLoss
  expr: rate(node_network_transmit_drop_total[5m]) > 0.01
  for: 5m
  annotations:
    summary: "High packet loss detected"

- alert: InterServiceConnectivity
  expr: probe_success{job="blackbox"} == 0
  for: 2m
  annotations:
    summary: "Service connectivity lost"
```

### Grafana Dashboards

Monitor:
1. **Network I/O**: Bytes sent/received
2. **Connection Status**: Active connections per service
3. **Latency**: Request duration histograms
4. **Error Rate**: Failed connections / total connections
5. **Geographic Map**: Multi-region connectivity status

## Prevention Strategies

### 1. Network Redundancy

```yaml
# Multiple network paths
# Docker networks with redundancy
networks:
  hololoom-primary:
    driver: bridge
  hololoom-backup:
    driver: overlay  # For multi-host
```

### 2. Health Checks

```yaml
# In docker-compose.voice.yml
services:
  voice-agent:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 3. Automatic Failover

```python
# Failover manager with aggressive health checks
from HoloLoom.voice.failover import FailoverManager, FailoverConfig

config = FailoverConfig(
    health_check_interval=10.0,  # Check every 10s
    down_threshold_failures=2,   # Fail after 2 failures
    enable_auto_failback=True
)
```

### 4. Connection Pooling

```python
# Maintain persistent connections
# Retry with exponential backoff
# Circuit breaker pattern
```

### 5. Multi-Region Deployment

Deploy to multiple availability zones and regions:
- **Primary**: us-east-1a, us-east-1b
- **Secondary**: us-west-2a, us-west-2b
- **Tertiary**: eu-west-1a (optional)

## Common Network Issues

### Issue: DNS Resolution Failure

```bash
# Symptoms
curl: (6) Could not resolve host

# Fix
echo "nameserver 8.8.8.8" >> /etc/resolv.conf
systemctl restart systemd-resolved
```

### Issue: Docker Network Corruption

```bash
# Symptoms
Error response from daemon: network not found

# Fix
docker network prune -f
docker-compose down && docker-compose up -d
```

### Issue: Firewall Blocking

```bash
# Symptoms
Connection timeout

# Fix - allow Docker networks
iptables -A INPUT -i docker0 -j ACCEPT
ufw allow from 172.17.0.0/16
```

### Issue: MTU Mismatch

```bash
# Symptoms
Packet fragmentation, slow connections

# Fix
ip link set dev eth0 mtu 1500
docker network create --opt com.docker.network.driver.mtu=1500 hololoom-network
```

## Escalation

Escalate if:
- Partition persists > 15 minutes
- Multiple regions affected
- Underlying cloud provider issue
- Repeated partitions (>3/day)

**Contact**:
- **Network Team**: network-oncall@hololoom.ai
- **Cloud Provider**: AWS Support (Premium)
- **Slack**: #incidents-network

## References

- [Docker Networking](https://docs.docker.com/network/)
- [Neo4j Cluster Documentation](https://neo4j.com/docs/operations-manual/current/clustering/)
- [AWS Network Troubleshooting](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-troubleshooting.html)
- [Failover Manager Implementation](../../HoloLoom/voice/failover.py)

---

**Last Reviewed**: 2025-11-16
**Reviewer**: Agent H - Wave 3 Production Hardening
**Next Review**: 2025-12-16
