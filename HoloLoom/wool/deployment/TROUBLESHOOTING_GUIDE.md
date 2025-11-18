# Wool Storage Troubleshooting Guide

Comprehensive troubleshooting guide for diagnosing and resolving issues in Wool Storage clusters.

**Last Updated**: November 18, 2025
**Version**: 1.0

---

## Table of Contents

1. [Quick Diagnostic Commands](#quick-diagnostic-commands)
2. [Pod Issues](#pod-issues)
3. [Cluster Formation Issues](#cluster-formation-issues)
4. [Performance Issues](#performance-issues)
5. [Replication Issues](#replication-issues)
6. [Storage Issues](#storage-issues)
7. [Network Issues](#network-issues)
8. [Data Consistency Issues](#data-consistency-issues)
9. [Version & Merge Issues](#version--merge-issues)
10. [Resource Exhaustion](#resource-exhaustion)
11. [Monitoring & Metrics Issues](#monitoring--metrics-issues)
12. [Recovery Procedures](#recovery-procedures)

---

## Quick Diagnostic Commands

### Check Overall Health

```bash
# Kubernetes: Pod status
kubectl get pods -n wool-storage -o wide

# Kubernetes: Recent events
kubectl get events -n wool-storage --sort-by='.lastTimestamp' | tail -20

# Docker Compose: Service status
docker-compose ps

# Docker Compose: Recent logs
docker-compose logs --tail=50 --follow

# Check node health endpoint
curl http://localhost:9000/health | jq '.'

# Check cluster status
curl http://localhost:9000/stats | jq '.cluster'

# Check replication status
curl http://localhost:9000/stats | jq '.replication'

# Prometheus metrics
curl http://localhost:9000/metrics | grep wool_
```

### Quick Status Script

```bash
#!/bin/bash
# quick-status.sh - Check wool storage cluster status

echo "=== Pod Status ==="
kubectl get pods -n wool-storage

echo -e "\n=== Service Status ==="
kubectl get svc -n wool-storage

echo -e "\n=== Node Health ==="
for i in {0..2}; do
  echo "Node wool-storage-$i:"
  kubectl exec wool-storage-$i -n wool-storage -- curl -s localhost:9000/health | jq '.healthy'
done

echo -e "\n=== Cluster Stats ==="
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/stats | jq '.cluster'

echo -e "\n=== Recent Errors ==="
kubectl logs -n wool-storage --selector=app=wool-storage --tail=20 | grep -i error
```

---

## Pod Issues

### Pod Not Starting

**Symptoms**:
- Pod stuck in `Pending`, `ContainerCreating`, or `CrashLoopBackOff`
- `kubectl get pods` shows non-Running status

**Diagnosis**:

```bash
# Check pod status and events
kubectl describe pod wool-storage-0 -n wool-storage

# Check pod logs
kubectl logs wool-storage-0 -n wool-storage

# Check previous logs (if container restarted)
kubectl logs wool-storage-0 -n wool-storage --previous

# Check resource availability
kubectl get nodes -o wide
kubectl describe node <node-name>
```

**Common Causes & Solutions**:

#### 1. PVC Not Bound

```bash
# Check PVC status
kubectl get pvc -n wool-storage

# If PVC stuck in Pending:
kubectl describe pvc data-wool-storage-0 -n wool-storage
```

**Solution**:
- Ensure StorageClass exists: `kubectl get sc`
- Provision storage manually or fix dynamic provisioner
- Check node has available disk space

#### 2. Image Pull Failure

```bash
# Check image pull errors
kubectl describe pod wool-storage-0 -n wool-storage | grep -A5 "Events:"
```

**Solution**:
- Verify image exists: `docker pull wool-storage:latest`
- Check image registry credentials: `kubectl get secrets -n wool-storage`
- Use correct image name in StatefulSet

#### 3. Resource Limits

```bash
# Check node capacity
kubectl describe node <node-name> | grep -A10 "Allocated resources"
```

**Solution**:
- Reduce resource requests/limits in StatefulSet
- Add more nodes to cluster
- Evict non-critical pods

#### 4. Init Container Failure

```bash
# Check init container logs
kubectl logs wool-storage-0 -n wool-storage -c <init-container-name>
```

**Solution**:
- Fix init container script errors
- Ensure dependencies are available
- Check permissions

### Pod Crashing

**Symptoms**:
- Pod in `CrashLoopBackOff` state
- Container restarts repeatedly

**Diagnosis**:

```bash
# Check crash logs
kubectl logs wool-storage-0 -n wool-storage --previous | tail -100

# Check exit code
kubectl describe pod wool-storage-0 -n wool-storage | grep "Exit Code"

# Check liveness/readiness probes
kubectl describe pod wool-storage-0 -n wool-storage | grep -A10 "Liveness"
```

**Common Exit Codes**:
- **Exit 0**: Clean shutdown (expected during rolling updates)
- **Exit 1**: Application error (check logs for stack trace)
- **Exit 137**: Killed by OOM (out of memory) - increase memory limits
- **Exit 139**: Segmentation fault (corrupted data or bug)
- **Exit 143**: Terminated by SIGTERM (graceful shutdown)

**Solutions by Exit Code**:

```yaml
# Exit 137 (OOM): Increase memory
resources:
  limits:
    memory: "8Gi"  # Increase from 4Gi

# Exit 1 (Application error): Check logs
kubectl logs wool-storage-0 -n wool-storage --previous | grep -i error

# Liveness probe failure: Increase timeout
livenessProbe:
  initialDelaySeconds: 60  # Increase from 30
  timeoutSeconds: 10       # Increase from 5
```

### Pod Evicted

**Symptoms**:
- Pod status: `Evicted`
- Pod disappears and recreates

**Diagnosis**:

```bash
# Check eviction reason
kubectl get pod wool-storage-0 -n wool-storage -o yaml | grep -A5 "reason: Evicted"

# Check node pressure
kubectl describe node <node-name> | grep -i pressure
```

**Common Causes**:
- **DiskPressure**: Node disk >85% full
- **MemoryPressure**: Node memory exhausted
- **NodePressure**: CPU or other resources

**Solutions**:

```bash
# DiskPressure: Clean up old images
kubectl exec <node> -- docker image prune -a

# MemoryPressure: Adjust limits or add nodes
kubectl top nodes
kubectl scale deployment <low-priority> --replicas=0 -n <namespace>

# Prevent eviction with PodDisruptionBudget (already configured):
kubectl get pdb -n wool-storage
```

---

## Cluster Formation Issues

### Nodes Not Discovering Each Other

**Symptoms**:
- Single-node clusters instead of 3-node cluster
- `wool_cluster_nodes_total` = 1 per node (should be 3)

**Diagnosis**:

```bash
# Check gossip membership
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/stats | jq '.cluster.members'

# Check DNS resolution
kubectl exec wool-storage-0 -n wool-storage -- nslookup wool-storage-1.wool-storage-headless.wool-storage.svc.cluster.local

# Check PEERS environment variable
kubectl exec wool-storage-0 -n wool-storage -- env | grep PEERS

# Check network connectivity
kubectl exec wool-storage-0 -n wool-storage -- ping wool-storage-1.wool-storage-headless
```

**Common Causes & Solutions**:

#### 1. DNS Resolution Failure

**Solution**:
```bash
# Verify headless service exists
kubectl get svc wool-storage-headless -n wool-storage

# Check service endpoints
kubectl get endpoints wool-storage-headless -n wool-storage

# If missing, recreate service:
kubectl delete svc wool-storage-headless -n wool-storage
kubectl apply -f wool-statefulset.yaml
```

#### 2. Incorrect PEERS Configuration

**Solution**:
```yaml
# Fix PEERS environment variable in StatefulSet
env:
  - name: PEERS
    value: "wool-storage-0.wool-storage-headless.wool-storage.svc.cluster.local,wool-storage-1.wool-storage-headless.wool-storage.svc.cluster.local,wool-storage-2.wool-storage-headless.wool-storage.svc.cluster.local"
```

#### 3. Gossip Port Blocked

**Solution**:
```bash
# Check NetworkPolicy
kubectl get networkpolicy -n wool-storage

# Test port connectivity
kubectl exec wool-storage-0 -n wool-storage -- nc -zv wool-storage-1 9000

# If blocked, create NetworkPolicy to allow:
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: wool-storage-gossip
spec:
  podSelector:
    matchLabels:
      app: wool-storage
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: wool-storage
      ports:
        - protocol: TCP
          port: 9000
```

### Split Brain

**Symptoms**:
- Cluster splits into multiple independent clusters
- Conflicting data across nodes
- `wool_cluster_nodes_healthy` < `wool_cluster_nodes_total`

**Diagnosis**:

```bash
# Check membership on each node
for i in {0..2}; do
  echo "Node $i members:"
  kubectl exec wool-storage-$i -n wool-storage -- curl -s localhost:9000/stats | jq '.cluster.members'
done

# Check partition detection
kubectl logs -n wool-storage --selector=app=wool-storage | grep -i "partition"
```

**Causes**:
- Network partition between nodes
- Gossip protocol timeout too aggressive
- Simultaneous node failures

**Solutions**:

```bash
# 1. Restart minority partition (force rejoin)
kubectl delete pod wool-storage-2 -n wool-storage

# 2. If severe, restart all nodes sequentially
for i in {0..2}; do
  kubectl delete pod wool-storage-$i -n wool-storage
  sleep 30  # Wait for rejoin
done

# 3. Increase gossip timeout in config:
# wool_storage.yaml
gossip:
  probe_interval: 5s        # Increase from 1s
  probe_timeout: 10s        # Increase from 3s
  suspicion_mult: 4         # Increase from 3
```

**Prevention**:
- Use anti-affinity rules (already configured)
- Ensure stable network
- Monitor `wool_replication_lag_seconds`

---

## Performance Issues

### High Latency

**Symptoms**:
- `wool_storage_latency_seconds{quantile="0.95"} > 1.0`
- Slow client operations

**Diagnosis**:

```bash
# Check latency metrics
curl http://localhost:9000/metrics | grep wool_storage_latency

# Check resource usage
kubectl top pods -n wool-storage

# Check disk I/O
kubectl exec wool-storage-0 -n wool-storage -- iostat -x 1 5

# Check network latency
kubectl exec wool-storage-0 -n wool-storage -- ping -c 10 wool-storage-1
```

**Common Causes & Solutions**:

#### 1. Disk I/O Bottleneck

**Diagnosis**:
```bash
# Check I/O wait
kubectl exec wool-storage-0 -n wool-storage -- top -bn1 | grep "Cpu"
# Look for high %wa (I/O wait)

# Check disk latency
curl http://localhost:9000/metrics | grep wool_disk_latency
```

**Solutions**:
- Use SSD storage class instead of HDD
- Increase IOPS limit for cloud disks (AWS EBS, GCE PD)
- Reduce concurrent operations

```yaml
# Use fast storage class
volumeClaimTemplates:
  spec:
    storageClassName: "fast-ssd"  # Instead of "standard"
```

#### 2. Network Latency

**Diagnosis**:
```bash
# Inter-node latency
for i in {1..2}; do
  kubectl exec wool-storage-0 -n wool-storage -- ping -c 10 wool-storage-$i | grep avg
done
```

**Solutions**:
- Use pod anti-affinity to co-locate on same node (trade availability for latency)
- Ensure nodes in same availability zone
- Increase network bandwidth (cloud instance type)

#### 3. CPU Saturation

**Diagnosis**:
```bash
# Check CPU throttling
kubectl top pods -n wool-storage
kubectl describe pod wool-storage-0 -n wool-storage | grep -A5 "Limits"
```

**Solutions**:
```yaml
# Increase CPU limits
resources:
  limits:
    cpu: "4000m"  # Increase from 2000m
```

#### 4. Compression Overhead

**Diagnosis**:
```bash
# Check compression CPU usage
curl http://localhost:9000/metrics | grep wool_compression_cpu_seconds
```

**Solutions**:
- Switch to LZ4 (faster) instead of Zstd (better compression)
- Disable compression for incompressible content (images, video)

```python
# Configuration
compression:
  algorithm: "lz4"  # Instead of "zstd"
  level: 1          # Fastest (instead of 3)
```

### Low Throughput

**Symptoms**:
- `rate(wool_storage_operations_total[1m]) < 100`
- Slow batch operations

**Diagnosis**:

```bash
# Check throughput
curl http://localhost:9000/metrics | grep -E "wool_storage_operations_total|wool_storage_errors_total"

# Check bottlenecks
kubectl top pods -n wool-storage
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/stats | jq '.'
```

**Solutions**:

1. **Parallelize operations**: Use batch API
2. **Increase workers**: Adjust concurrency settings
3. **Reduce replication factor**: 3x → 2x (trades durability for performance)
4. **Use consistency level ONE**: Instead of QUORUM (faster writes)

```python
# Client configuration
client = WoolClient(
    consistency_level=ConsistencyLevel.ONE,  # Fastest
    batch_size=100,                           # Batch operations
    max_concurrent_requests=50                # Parallel requests
)
```

---

## Replication Issues

### Replication Lag

**Symptoms**:
- `wool_replication_lag_seconds > 10`
- Stale reads from replicas

**Diagnosis**:

```bash
# Check replication lag
curl http://localhost:9000/metrics | grep wool_replication_lag_seconds

# Check replication queue depth
curl http://localhost:9000/stats | jq '.replication.pending_count'

# Check failed replications
curl http://localhost:9000/metrics | grep wool_replication_errors_total
```

**Causes**:
- Slow replica nodes (CPU, disk, network)
- Replication queue overflow
- Network partition

**Solutions**:

```bash
# 1. Force replication catch-up
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST localhost:9000/admin/force-replicate

# 2. Increase replication workers
# In config:
replication:
  workers: 10  # Increase from 5
  queue_size: 10000  # Increase buffer

# 3. Reduce write load temporarily
# Throttle clients or use consistency level ONE

# 4. Check replica health
kubectl exec wool-storage-1 -n wool-storage -- curl -s localhost:9000/health
```

### Under-Replicated Data

**Symptoms**:
- `wool_replication_factor < 3`
- Data loss risk warnings

**Diagnosis**:

```bash
# Check replication status
curl http://localhost:9000/stats | jq '.replication'

# Check which files are under-replicated
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/admin/under-replicated | jq '.'
```

**Solutions**:

```bash
# 1. Trigger re-replication
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST localhost:9000/admin/re-replicate

# 2. Check for failed nodes
kubectl get pods -n wool-storage | grep -v Running

# 3. Restore failed node
kubectl delete pod wool-storage-<failed> -n wool-storage

# 4. Monitor progress
watch 'curl -s http://localhost:9000/stats | jq ".replication.under_replicated_count"'
```

---

## Storage Issues

### Disk Space Full

**Symptoms**:
- `wool_storage_bytes_total / wool_storage_capacity_bytes > 0.95`
- Write operations fail with "No space left on device"

**Diagnosis**:

```bash
# Check disk usage
kubectl exec wool-storage-0 -n wool-storage -- df -h /data

# Check largest files
kubectl exec wool-storage-0 -n wool-storage -- du -h /data | sort -rh | head -20

# Check PVC size
kubectl get pvc -n wool-storage
```

**Solutions**:

#### 1. Expand PVC (if supported by StorageClass)

```bash
# Check if storage class allows expansion
kubectl get sc <storage-class> -o yaml | grep allowVolumeExpansion

# Expand PVC
kubectl patch pvc data-wool-storage-0 -n wool-storage -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'

# Wait for expansion
kubectl get pvc data-wool-storage-0 -n wool-storage -w
```

#### 2. Enable Garbage Collection

```python
# Configure GC (future feature)
garbage_collection:
  enabled: true
  orphaned_versions_ttl: "7d"
  run_interval: "1h"
```

#### 3. Delete Old Versions

```bash
# Manual cleanup (use with caution!)
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST \
  -d '{"before": "2025-01-01T00:00:00Z"}' \
  localhost:9000/admin/delete-old-versions
```

#### 4. Add More Nodes (Scale Out)

```bash
# Scale StatefulSet
kubectl scale statefulset wool-storage --replicas=5 -n wool-storage

# Wait for data rebalancing
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/stats | jq '.rebalancing'
```

### Data Corruption

**Symptoms**:
- Checksum verification failures
- Unreadable files
- Crashes when accessing specific files

**Diagnosis**:

```bash
# Check for corruption errors
kubectl logs -n wool-storage --selector=app=wool-storage | grep -i "corrupt"

# Run integrity check
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST localhost:9000/admin/verify-integrity

# Check file checksums
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/admin/checksum/<file_id>
```

**Solutions**:

```bash
# 1. Restore from replica
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST \
  -d '{"file_id": "<corrupted-file>", "source_node": "wool-storage-1"}' \
  localhost:9000/admin/restore-from-replica

# 2. Restore from backup (Velero)
velero restore create --from-backup wool-storage-daily-20251118

# 3. If corruption widespread, restore entire PVC
kubectl delete pod wool-storage-0 -n wool-storage
# Restore PVC from snapshot
kubectl apply -f pvc-restore.yaml
```

**Prevention**:
- Enable checksums (default)
- Use ECC RAM for nodes
- Regular backups (Velero daily)
- Monitor `wool_checksum_failures_total`

---

## Network Issues

### Network Partition

**Symptoms**:
- Nodes can't communicate
- Cluster splits into multiple groups
- `wool_cluster_nodes_healthy` fluctuates

**Diagnosis**:

```bash
# Test connectivity between all nodes
for i in {0..2}; do
  for j in {0..2}; do
    kubectl exec wool-storage-$i -n wool-storage -- nc -zv wool-storage-$j 9000
  done
done

# Check network policies
kubectl get networkpolicy -n wool-storage

# Check for packet loss
kubectl exec wool-storage-0 -n wool-storage -- ping -c 100 wool-storage-1 | grep loss
```

**Solutions**:

```bash
# 1. Check CNI health (Calico, Flannel, etc.)
kubectl get pods -n kube-system | grep -i network

# 2. Restart CNI if needed
kubectl delete pod -n kube-system -l k8s-app=<cni-name>

# 3. Check cloud firewall rules (AWS Security Groups, GCP Firewall)
# Ensure port 9000 is open between nodes

# 4. Temporarily disable NetworkPolicy for testing
kubectl delete networkpolicy -n wool-storage --all
```

### High Network Latency

**Symptoms**:
- Slow replication
- High `wool_network_latency_seconds`

**Diagnosis**:

```bash
# Measure inter-node latency
kubectl exec wool-storage-0 -n wool-storage -- ping -c 100 wool-storage-1 | grep avg

# Check network bandwidth
kubectl exec wool-storage-0 -n wool-storage -- iperf3 -c wool-storage-1
```

**Solutions**:
- Use pod anti-affinity to place on same node (if latency critical)
- Deploy in same availability zone
- Increase network bandwidth (upgrade instance type)
- Enable compression for replication (if not CPU-bound)

---

## Data Consistency Issues

### Read-After-Write Inconsistency

**Symptoms**:
- Write succeeds but immediate read returns old data
- Different results from different nodes

**Diagnosis**:

```bash
# Test consistency
# Write to node 0
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST -d '{"data":"test"}' localhost:9000/store/test-file

# Read from node 1 immediately
kubectl exec wool-storage-1 -n wool-storage -- curl localhost:9000/read/test-file

# Check consistency level used
kubectl logs wool-storage-0 -n wool-storage | grep "consistency_level"
```

**Causes**:
- Using consistency level ONE (fastest, least consistent)
- Replication lag
- Clock skew

**Solutions**:

```python
# Use stronger consistency
client = WoolClient(
    consistency_level=ConsistencyLevel.QUORUM  # Read/write majority
)

# Or use ALL for strongest consistency (slowest)
client = WoolClient(
    consistency_level=ConsistencyLevel.ALL
)
```

### Version Conflicts

**Symptoms**:
- Merge conflicts on concurrent writes
- `wool_merge_conflicts_total` increasing

**Diagnosis**:

```bash
# Check conflict rate
curl http://localhost:9000/metrics | grep wool_merge_conflicts_total

# Get conflicted files
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/admin/conflicts | jq '.'
```

**Solutions**:

```bash
# 1. Resolve conflicts manually
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST \
  -d '{"file_id": "<file>", "resolution": "ours"}' \
  localhost:9000/admin/resolve-conflict

# 2. Use optimistic locking in clients
# Include version in writes:
client.write(file_id="test", data=data, expected_version=5)

# 3. Reduce concurrent writes to same file
# Use application-level locking or queues
```

---

## Version & Merge Issues

### Merge Failures

**Symptoms**:
- Merge operations fail
- `wool_merge_errors_total` increasing

**Diagnosis**:

```bash
# Check merge errors
curl http://localhost:9000/metrics | grep wool_merge_errors_total

# Get failed merges
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/admin/failed-merges | jq '.'

# Check logs for errors
kubectl logs -n wool-storage --selector=app=wool-storage | grep -i "merge.*error"
```

**Solutions**:

```bash
# 1. Retry merge with conflict resolution
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST \
  -d '{"source_branch": "feature", "target_branch": "main", "strategy": "ours"}' \
  localhost:9000/merge

# 2. Manual 3-way merge
# Download conflicting versions
curl http://localhost:9000/version/<file>?version=<v1> > v1.txt
curl http://localhost:9000/version/<file>?version=<v2> > v2.txt
# Resolve manually, then create merge commit

# 3. Abandon and recreate branch
kubectl exec wool-storage-0 -n wool-storage -- curl -X DELETE localhost:9000/branch/feature
```

### Time-Travel Query Failures

**Symptoms**:
- Historical queries return errors
- `wool_time_travel_errors_total` increasing

**Diagnosis**:

```bash
# Test time-travel query
kubectl exec wool-storage-0 -n wool-storage -- curl \
  "localhost:9000/read/test-file?at=2025-01-01T00:00:00Z"

# Check version chain integrity
kubectl exec wool-storage-0 -n wool-storage -- curl -s \
  localhost:9000/admin/verify-version-chain/<file_id> | jq '.'
```

**Solutions**:

```bash
# 1. Rebuild version index
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST \
  localhost:9000/admin/rebuild-version-index

# 2. Restore missing versions from backup
velero restore create --from-backup wool-storage-weekly-20251101 \
  --include-namespaces wool-storage \
  --selector="wool.storage/file-id=<file_id>"

# 3. If version chain broken, recreate from snapshots
# (manual data recovery procedure)
```

---

## Resource Exhaustion

### Out of Memory (OOM)

**Symptoms**:
- Pod killed with exit code 137
- `process_resident_memory_bytes` near limit

**Diagnosis**:

```bash
# Check memory usage
kubectl top pods -n wool-storage

# Check memory limits
kubectl describe pod wool-storage-0 -n wool-storage | grep -A5 "Limits"

# Check OOM kills
kubectl get events -n wool-storage | grep OOM
```

**Solutions**:

```yaml
# 1. Increase memory limits
resources:
  limits:
    memory: "8Gi"  # Increase from 4Gi

# 2. Enable memory profiling to find leaks
# Add to container args:
args:
  - "--enable-pprof"
  - "--pprof-addr=:6060"

# Profile memory:
kubectl port-forward wool-storage-0 6060:6060 -n wool-storage
go tool pprof http://localhost:6060/debug/pprof/heap
```

### CPU Throttling

**Symptoms**:
- High latency despite low load
- `container_cpu_cfs_throttled_seconds_total` increasing

**Diagnosis**:

```bash
# Check CPU throttling
kubectl exec wool-storage-0 -n wool-storage -- cat /sys/fs/cgroup/cpu/cpu.stat | grep throttled

# Check CPU usage vs limit
kubectl top pods -n wool-storage
```

**Solutions**:

```yaml
# Increase CPU limits
resources:
  limits:
    cpu: "4000m"  # Increase from 2000m

# Or remove limits (not recommended for production)
resources:
  limits:
    cpu: null  # No limit (use with caution)
```

### File Descriptor Exhaustion

**Symptoms**:
- "Too many open files" errors
- `process_open_fds` near limit

**Diagnosis**:

```bash
# Check open file descriptors
kubectl exec wool-storage-0 -n wool-storage -- ls -l /proc/self/fd | wc -l

# Check limit
kubectl exec wool-storage-0 -n wool-storage -- ulimit -n
```

**Solutions**:

```yaml
# Increase file descriptor limit in pod security context
securityContext:
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  sysctls:
    - name: fs.file-max
      value: "65536"
```

---

## Monitoring & Metrics Issues

### Metrics Not Exported

**Symptoms**:
- Prometheus shows no data
- `/metrics` endpoint returns 404

**Diagnosis**:

```bash
# Check metrics endpoint
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/metrics | head -20

# Check Prometheus scraping
kubectl logs -n wool-storage prometheus-<pod> | grep -i wool

# Check Prometheus targets
kubectl port-forward -n wool-storage svc/prometheus 9090:9090
# Open http://localhost:9090/targets
```

**Solutions**:

```bash
# 1. Verify metrics endpoint enabled
# In config:
metrics:
  enabled: true
  port: 9000
  path: "/metrics"

# 2. Check Prometheus service discovery
kubectl get servicemonitor -n wool-storage

# 3. Check network access
kubectl exec -n wool-storage prometheus-<pod> -- curl wool-storage-0:9000/metrics
```

### Dashboard Shows "No Data"

**Diagnosis**:

```bash
# Check Grafana datasource
# In Grafana UI: Configuration → Data Sources → Prometheus → Test

# Check PromQL queries directly
kubectl port-forward -n wool-storage svc/prometheus 9090:9090
# Open http://localhost:9090
# Run query: wool_storage_operations_total
```

**Solutions**:
- Fix Prometheus datasource URL in Grafana
- Import dashboards with correct UID
- Verify metric names match (check for typos)

---

## Recovery Procedures

### Complete Cluster Failure

**Scenario**: All 3 nodes down simultaneously

**Recovery Steps**:

```bash
# 1. Don't panic - data is on persistent volumes

# 2. Check PVCs still exist
kubectl get pvc -n wool-storage

# 3. Restart StatefulSet
kubectl delete statefulset wool-storage -n wool-storage
kubectl apply -f wool-statefulset.yaml

# 4. Wait for pods to start sequentially
kubectl get pods -n wool-storage -w

# 5. Verify cluster reforms
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/stats | jq '.cluster.members'

# 6. Verify data integrity
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST localhost:9000/admin/verify-integrity
```

### Data Recovery from Backup

**Using Velero**:

```bash
# 1. List available backups
velero backup get

# 2. Restore from specific backup
velero restore create wool-recovery \
  --from-backup wool-storage-weekly-20251101 \
  --include-namespaces wool-storage

# 3. Monitor restore progress
velero restore describe wool-recovery

# 4. Verify data
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/stats | jq '.files_total'
```

### Node Data Corruption

**Scenario**: One node has corrupted data

**Recovery Steps**:

```bash
# 1. Identify corrupted node
kubectl exec wool-storage-1 -n wool-storage -- curl -X POST localhost:9000/admin/verify-integrity
# Returns: {"healthy": false, "corrupted_files": 15}

# 2. Delete pod and PVC (data loss on this node only)
kubectl delete pod wool-storage-1 -n wool-storage
kubectl delete pvc data-wool-storage-1 -n wool-storage

# 3. Wait for pod recreation with fresh PVC
kubectl get pods -n wool-storage -w

# 4. Trigger re-replication from healthy nodes
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST localhost:9000/admin/re-replicate

# 5. Monitor progress
watch 'kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/stats | jq ".replication"'
```

---

## Support & Escalation

### Collecting Debug Information

```bash
#!/bin/bash
# collect-debug-info.sh - Gather diagnostic information

DEBUG_DIR="wool-debug-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEBUG_DIR"

echo "Collecting debug information..."

# Pod status
kubectl get pods -n wool-storage -o wide > "$DEBUG_DIR/pods.txt"

# Describe pods
for i in {0..2}; do
  kubectl describe pod wool-storage-$i -n wool-storage > "$DEBUG_DIR/describe-pod-$i.txt"
done

# Logs
for i in {0..2}; do
  kubectl logs wool-storage-$i -n wool-storage --tail=1000 > "$DEBUG_DIR/logs-$i.txt"
  kubectl logs wool-storage-$i -n wool-storage --previous --tail=1000 > "$DEBUG_DIR/logs-$i-previous.txt" 2>/dev/null || true
done

# Events
kubectl get events -n wool-storage --sort-by='.lastTimestamp' > "$DEBUG_DIR/events.txt"

# Cluster stats
for i in {0..2}; do
  kubectl exec wool-storage-$i -n wool-storage -- curl -s localhost:9000/stats > "$DEBUG_DIR/stats-$i.json"
done

# Metrics
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/metrics > "$DEBUG_DIR/metrics.txt"

# Resource usage
kubectl top pods -n wool-storage > "$DEBUG_DIR/top.txt"

# PVCs
kubectl get pvc -n wool-storage -o yaml > "$DEBUG_DIR/pvcs.yaml"

# ConfigMaps & Secrets
kubectl get cm -n wool-storage -o yaml > "$DEBUG_DIR/configmaps.yaml"

tar czf "$DEBUG_DIR.tar.gz" "$DEBUG_DIR"
rm -rf "$DEBUG_DIR"

echo "Debug information collected: $DEBUG_DIR.tar.gz"
```

### Where to Get Help

1. **Documentation**:
   - [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
   - [Grafana Dashboards](dashboards/README.md)
   - [Performance Tuning Guide](PERFORMANCE_TUNING_GUIDE.md)

2. **Metrics & Dashboards**:
   - Grafana: http://localhost:3000
   - Prometheus: http://localhost:9090

3. **GitHub Issues**:
   - https://github.com/yourusername/hololoom/issues
   - Include debug bundle: `collect-debug-info.sh`

4. **Community**:
   - Slack: #wool-storage
   - Discord: Wool Storage Support

---

**Author**: Claude Code
**Date**: November 18, 2025
**Status**: Production Ready
**Version**: 1.0
