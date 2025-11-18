# Wool Storage Migration Guide

Complete guide for migrating from single-node to distributed cluster, and performing zero-downtime upgrades.

**Last Updated**: November 18, 2025
**Version**: 1.0

---

## Table of Contents

1. [Single-Node to Distributed Cluster](#single-node-to-distributed-cluster)
2. [Zero-Downtime Upgrades](#zero-downtime-upgrades)
3. [Cross-Region Migration](#cross-region-migration)
4. [Storage Backend Migration](#storage-backend-migration)
5. [Data Export/Import](#data-exportimport)
6. [Rollback Procedures](#rollback-procedures)

---

## Single-Node to Distributed Cluster

### Overview

Migrating from a single-node deployment to a 3-node distributed cluster for:
- **High availability**: Survive node failures
- **Horizontal scaling**: Handle more load
- **Data durability**: 3x replication

**Downtime**: 5-10 minutes (configurable with read-only mode)

### Prerequisites

- **Existing single-node deployment** with data
- **Kubernetes cluster** with 3+ nodes
- **Persistent storage** (PVCs) provisioned
- **Backup** of existing data (critical!)

### Migration Strategy

We'll use a **"scale-out with data copy"** approach:

1. Create backup of single-node data
2. Deploy 3-node cluster
3. Copy data to first node (seed node)
4. Let cluster auto-replicate (3x)
5. Verify data integrity
6. Switch traffic to new cluster

---

### Step 1: Backup Single-Node Data

**Using Velero (Recommended)**:

```bash
# Install Velero if not already installed
velero install \
  --provider aws \
  --bucket wool-backups \
  --backup-location-config region=us-west-2 \
  --snapshot-location-config region=us-west-2 \
  --secret-file ./credentials-velero

# Create full backup
velero backup create wool-single-node-backup \
  --include-namespaces wool-storage \
  --wait

# Verify backup
velero backup describe wool-single-node-backup
```

**Manual Backup (Alternative)**:

```bash
# Create backup directory
BACKUP_DIR="/tmp/wool-backup-$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Copy data from single-node pod
kubectl cp wool-storage-single-0:/data "$BACKUP_DIR/data" -n wool-storage

# Verify backup
du -sh "$BACKUP_DIR"
ls -lR "$BACKUP_DIR/data" | wc -l

# Compress for transfer
tar czf wool-backup.tar.gz -C "$BACKUP_DIR" .

# Upload to S3 or storage
aws s3 cp wool-backup.tar.gz s3://wool-backups/single-node-$(date +%Y%m%d).tar.gz
```

---

### Step 2: Deploy 3-Node Cluster

**Option A: Kubernetes StatefulSet**

```bash
# Create namespace (if different from single-node)
kubectl create namespace wool-storage-cluster

# Deploy 3-node cluster
kubectl apply -f wool-statefulset.yaml -n wool-storage-cluster

# Wait for pods to start (empty initially)
kubectl get pods -n wool-storage-cluster -w

# Verify cluster formation
kubectl exec wool-storage-0 -n wool-storage-cluster -- curl -s localhost:9000/stats | jq '.cluster.members'
# Should show 3 nodes
```

**Option B: Docker Compose (Development)**

```bash
# Copy docker-compose.yml to new directory
mkdir wool-cluster
cd wool-cluster
cp ../wool-deployment/docker-compose.yml .

# Start cluster
docker-compose up -d

# Verify cluster
docker-compose ps
docker-compose logs wool-1 | grep "Cluster formed"
```

---

### Step 3: Copy Data to Seed Node

**Kubernetes Method**:

```bash
# Extract backup to seed node (wool-storage-0)
# If using Velero:
velero restore create wool-seed-restore \
  --from-backup wool-single-node-backup \
  --namespace-mappings wool-storage:wool-storage-cluster

# If using manual backup:
# Upload backup to pod
kubectl cp wool-backup.tar.gz wool-storage-0:/tmp/ -n wool-storage-cluster

# Extract inside pod
kubectl exec wool-storage-0 -n wool-storage-cluster -- tar xzf /tmp/wool-backup.tar.gz -C /data

# Verify data
kubectl exec wool-storage-0 -n wool-storage-cluster -- ls -lR /data | wc -l

# Check file count
kubectl exec wool-storage-0 -n wool-storage-cluster -- curl -s localhost:9000/stats | jq '.files_total'
```

**Docker Compose Method**:

```bash
# Copy data to first node volume
docker cp wool-backup.tar.gz wool-node-1:/data/
docker exec wool-node-1 tar xzf /data/wool-backup.tar.gz -C /data/

# Verify
docker exec wool-node-1 ls -lR /data | wc -l
```

---

### Step 4: Trigger Auto-Replication

**Force Full Replication**:

```bash
# Trigger replication from seed node to replicas
kubectl exec wool-storage-0 -n wool-storage-cluster -- curl -X POST \
  -d '{"force_full_sync": true}' \
  localhost:9000/admin/replicate-all

# Monitor replication progress
watch 'kubectl exec wool-storage-0 -n wool-storage-cluster -- curl -s localhost:9000/stats | jq ".replication"'

# Expected output:
# {
#   "pending_count": 1500,         # Decreasing
#   "replicated_count": 3500,      # Increasing
#   "under_replicated_count": 800, # Should reach 0
#   "replication_factor": 3.0      # Should reach 3.0
# }
```

**Monitor with Grafana**:

```bash
# Port-forward Grafana
kubectl port-forward svc/grafana 3000:3000 -n wool-storage-cluster

# Open http://localhost:3000
# Dashboard: Cluster Overview
# Panel: Replication Factor (should reach 3.0)
# Panel: Under-Replicated Files (should reach 0)
```

**Replication typically takes**:
- **Small datasets (<10GB)**: 5-15 minutes
- **Medium datasets (10-100GB)**: 30-60 minutes
- **Large datasets (>100GB)**: 1-2 hours

**Bottlenecks**:
- Network bandwidth between nodes
- Disk I/O on replica nodes
- CPU for compression/decompression

---

### Step 5: Verify Data Integrity

**Run Integrity Check**:

```bash
# Check all nodes
for i in {0..2}; do
  echo "Verifying node $i..."
  kubectl exec wool-storage-$i -n wool-storage-cluster -- curl -X POST \
    localhost:9000/admin/verify-integrity | jq '.'
done

# Expected output per node:
# {
#   "healthy": true,
#   "total_files": 5000,
#   "verified_files": 5000,
#   "corrupted_files": 0,
#   "checksum_failures": 0
# }
```

**Spot-Check Critical Files**:

```bash
# List important files from old cluster
kubectl exec wool-storage-single-0 -n wool-storage -- curl -s localhost:9000/list | jq '.files[0:10]'

# Verify same files exist in new cluster with correct content
for file_id in $(cat important_files.txt); do
  echo "Checking $file_id..."

  # Read from old cluster
  OLD_HASH=$(kubectl exec wool-storage-single-0 -n wool-storage -- curl -s localhost:9000/read/$file_id | sha256sum)

  # Read from new cluster
  NEW_HASH=$(kubectl exec wool-storage-0 -n wool-storage-cluster -- curl -s localhost:9000/read/$file_id | sha256sum)

  if [ "$OLD_HASH" == "$NEW_HASH" ]; then
    echo "✓ $file_id matches"
  else
    echo "✗ $file_id MISMATCH"
  fi
done
```

**Verify Metadata**:

```bash
# Check version history preserved
kubectl exec wool-storage-0 -n wool-storage-cluster -- curl -s \
  localhost:9000/versions/<file_id> | jq '.versions | length'

# Check branches preserved
kubectl exec wool-storage-0 -n wool-storage-cluster -- curl -s \
  localhost:9000/branches | jq '.branches'
```

---

### Step 6: Switch Traffic

**DNS-Based Switchover** (Recommended):

```bash
# Update DNS or service to point to new cluster
# Old: wool-storage.example.com → wool-storage-single LoadBalancer
# New: wool-storage.example.com → wool-storage-cluster LoadBalancer

# If using Kubernetes Services:
kubectl patch service wool-storage-external -n wool-storage \
  -p '{"spec":{"selector":{"app":"wool-storage-cluster"}}}'

# Verify new endpoint
nslookup wool-storage.example.com
# Should return new cluster IPs
```

**Client Configuration Update**:

```python
# Old client configuration
client = WoolClient(endpoints=["wool-storage-single-0:9000"])

# New client configuration (3 endpoints)
client = WoolClient(endpoints=[
    "wool-storage-0.wool-storage-cluster:9000",
    "wool-storage-1.wool-storage-cluster:9000",
    "wool-storage-2.wool-storage-cluster:9000"
])

# Client automatically load-balances and fails over
```

**Gradual Traffic Migration** (Blue-Green):

```bash
# Week 1: 10% traffic to new cluster
kubectl patch service wool-storage-lb -n wool-storage \
  -p '{"spec":{"sessionAffinity":"ClientIP","sessionAffinityConfig":{"clientIP":{"timeoutSeconds":3600}}}}'

# Configure load balancer weights:
# Old cluster: 90%
# New cluster: 10%

# Week 2: 50% traffic
# Week 3: 100% traffic (full cutover)
```

---

### Step 7: Decommission Single-Node

**After 1-2 weeks of successful operation**:

```bash
# 1. Stop single-node cluster
kubectl scale deployment wool-storage-single --replicas=0 -n wool-storage

# 2. Verify no traffic
kubectl logs wool-storage-single-0 -n wool-storage | grep "request" | tail -100
# Should show no recent requests

# 3. Create final backup
velero backup create wool-single-node-final \
  --include-namespaces wool-storage

# 4. Delete single-node resources
kubectl delete namespace wool-storage

# 5. Archive backup
aws s3 cp \
  s3://wool-backups/wool-single-node-final-*.tar.gz \
  s3://wool-archives/$(date +%Y)/
```

---

## Zero-Downtime Upgrades

### Rolling Update Strategy

Wool Storage supports **rolling updates** with zero downtime using Kubernetes StatefulSet.

**Process**:
1. Update one pod at a time
2. Wait for pod to become healthy
3. Move to next pod
4. Repeat until all pods updated

**Compatibility**: Wool Storage maintains backward compatibility for N-1 versions (e.g., v1.1 nodes can join v1.0 cluster).

---

### Step-by-Step Rolling Update

**1. Verify Current Version**:

```bash
# Check current version
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/version | jq '.version'
# Output: "1.0.0"

# Check all pods
for i in {0..2}; do
  kubectl exec wool-storage-$i -n wool-storage -- curl -s localhost:9000/version
done
```

**2. Update Container Image**:

```bash
# Update StatefulSet with new image
kubectl set image statefulset/wool-storage \
  wool-storage=wool-storage:1.1.0 \
  -n wool-storage

# Or edit YAML
kubectl edit statefulset wool-storage -n wool-storage
# Change: image: wool-storage:1.1.0
```

**3. Monitor Rolling Update**:

```bash
# Watch pod updates
kubectl rollout status statefulset/wool-storage -n wool-storage

# Detailed monitoring
kubectl get pods -n wool-storage -w

# Expected sequence:
# 1. wool-storage-0 terminates → new pod starts
# 2. wool-storage-1 terminates → new pod starts
# 3. wool-storage-2 terminates → new pod starts
```

**4. Verify Health During Update**:

```bash
# Check cluster health (should remain 2-3 healthy nodes)
watch 'kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/stats | jq ".cluster.members_healthy"'

# Monitor metrics
kubectl port-forward svc/grafana 3000:3000 -n wool-storage
# Dashboard: Cluster Overview
# Panel: Healthy Nodes (should stay ≥ 2)
```

**5. Verify Version After Update**:

```bash
# Check all pods running new version
for i in {0..2}; do
  echo "Pod $i:"
  kubectl exec wool-storage-$i -n wool-storage -- curl -s localhost:9000/version | jq '.version'
done

# All should output: "1.1.0"
```

---

### Configuration Changes

**For configuration-only changes** (no code update):

```bash
# Update ConfigMap
kubectl edit configmap wool-storage-config -n wool-storage

# Rolling restart to pick up new config
kubectl rollout restart statefulset/wool-storage -n wool-storage

# Verify new config applied
kubectl exec wool-storage-0 -n wool-storage -- cat /etc/wool/config.yaml
```

---

### Canary Deployment

**For risky upgrades, test on single pod first**:

```bash
# 1. Manually update pod-0 only
kubectl delete pod wool-storage-0 -n wool-storage

# 2. Before it recreates, update StatefulSet with updateStrategy: OnDelete
kubectl patch statefulset wool-storage -n wool-storage -p \
  '{"spec":{"updateStrategy":{"type":"OnDelete"}}}'

# 3. Manually update image for pod-0
kubectl set image statefulset/wool-storage wool-storage=wool-storage:1.1.0-canary -n wool-storage

# 4. Delete pod-0 to trigger recreation
kubectl delete pod wool-storage-0 -n wool-storage

# 5. Monitor pod-0 for issues
kubectl logs wool-storage-0 -n wool-storage -f

# 6. If stable after 1 hour, update remaining pods
kubectl delete pod wool-storage-1 -n wool-storage
# Wait 15 minutes...
kubectl delete pod wool-storage-2 -n wool-storage
```

---

## Cross-Region Migration

### Overview

Migrating Wool Storage cluster from one region to another (e.g., us-west-2 → eu-central-1).

**Use Cases**:
- Data sovereignty requirements
- Latency optimization (move closer to users)
- Cost optimization (cheaper region)

**Downtime**: 10-30 minutes (depending on data size)

---

### Migration Strategy

**"Backup and Restore" Approach** (Simplest):

```bash
# 1. Create full backup in source region (us-west-2)
velero backup create wool-uswest2-final \
  --include-namespaces wool-storage \
  --storage-location aws-uswest2 \
  --wait

# 2. Configure Velero in target region (eu-central-1)
velero backup-location create aws-eucentral1 \
  --provider aws \
  --bucket wool-backups-eucentral1 \
  --config region=eu-central-1

# 3. Restore in target region
velero restore create wool-eucentral1-restore \
  --from-backup wool-uswest2-final \
  --restore-volumes=true

# 4. Update client endpoints
# Old: wool-storage.uswest2.example.com
# New: wool-storage.eucentral1.example.com

# 5. Decommission source region cluster
kubectl delete namespace wool-storage --context=uswest2
```

**"Dual-Write" Approach** (Zero Downtime):

```bash
# 1. Deploy new cluster in target region
kubectl apply -f wool-statefulset.yaml --context=eucentral1

# 2. Configure clients to dual-write (both regions)
# Application code:
primary_client = WoolClient(region="us-west-2")
secondary_client = WoolClient(region="eu-central-1")

def write_file(file_id, data):
    primary_client.write(file_id, data)
    secondary_client.write(file_id, data)  # Async background

# 3. Backfill historical data
# Run migration script to copy existing files to new region

# 4. Switch reads to new region (gradually)
# Week 1: 10% reads from eu-central-1
# Week 2: 50%
# Week 3: 100%

# 5. Stop dual-writes, use eu-central-1 only

# 6. Decommission us-west-2
```

---

## Storage Backend Migration

### Migrating Persistent Volumes

**Scenario**: Move from standard HDD to fast SSD storage class

```bash
# 1. Create new PVCs with SSD storage class
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-wool-storage-0-ssd
  namespace: wool-storage
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd  # New storage class
  resources:
    requests:
      storage: 100Gi
EOF

# Repeat for data-wool-storage-1-ssd, data-wool-storage-2-ssd

# 2. Copy data from old PVC to new PVC
kubectl run data-mover --image=ubuntu --restart=Never -n wool-storage -- \
  /bin/bash -c "
    apt-get update && apt-get install -y rsync
    rsync -avP /source/ /dest/
  " \
  --overrides='
  {
    "spec": {
      "volumes": [
        {"name": "source", "persistentVolumeClaim": {"claimName": "data-wool-storage-0"}},
        {"name": "dest", "persistentVolumeClaim": {"claimName": "data-wool-storage-0-ssd"}}
      ],
      "containers": [{
        "name": "data-mover",
        "image": "ubuntu",
        "volumeMounts": [
          {"name": "source", "mountPath": "/source"},
          {"name": "dest", "mountPath": "/dest"}
        ]
      }]
    }
  }'

# 3. Update StatefulSet to use new PVCs
kubectl patch statefulset wool-storage -n wool-storage -p \
  '{"spec":{"volumeClaimTemplates":[{"metadata":{"name":"data"},"spec":{"storageClassName":"fast-ssd"}}]}}'

# 4. Rolling restart pods to pick up new PVCs
kubectl delete pod wool-storage-0 -n wool-storage
# Wait for healthy...
kubectl delete pod wool-storage-1 -n wool-storage
# Wait for healthy...
kubectl delete pod wool-storage-2 -n wool-storage
```

---

## Data Export/Import

### Export Data to External Format

**Export to tar.gz archive**:

```bash
# Export all data from cluster
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST \
  -d '{"format": "tar", "compression": "gzip"}' \
  localhost:9000/admin/export > wool-export.tar.gz

# Verify export
tar tzf wool-export.tar.gz | head -20

# Export specific file IDs
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST \
  -d '{"file_ids": ["file1", "file2"], "format": "tar"}' \
  localhost:9000/admin/export > partial-export.tar
```

**Export to S3**:

```python
# Python script for S3 export
import boto3
from wool_client import WoolClient

client = WoolClient(endpoint="wool-storage-0:9000")
s3 = boto3.client('s3')

# List all files
files = client.list_files()

# Export each to S3
for file_info in files:
    file_id = file_info['id']
    data = client.read(file_id)

    s3.put_object(
        Bucket='wool-exports',
        Key=f'files/{file_id}',
        Body=data
    )

    print(f"Exported {file_id}")
```

### Import Data from External Source

**Import from tar.gz**:

```bash
# Upload archive to pod
kubectl cp wool-import.tar.gz wool-storage-0:/tmp/ -n wool-storage

# Import via API
kubectl exec wool-storage-0 -n wool-storage -- curl -X POST \
  -F "file=@/tmp/wool-import.tar.gz" \
  localhost:9000/admin/import

# Monitor import progress
kubectl exec wool-storage-0 -n wool-storage -- curl -s \
  localhost:9000/admin/import-status | jq '.'
```

**Import from S3**:

```python
# Python script for S3 import
import boto3
from wool_client import WoolClient

client = WoolClient(endpoint="wool-storage-0:9000")
s3 = boto3.client('s3')

# List S3 objects
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket='wool-exports', Prefix='files/')

# Import each file
for page in pages:
    for obj in page.get('Contents', []):
        key = obj['Key']
        file_id = key.split('/')[-1]

        # Download from S3
        data = s3.get_object(Bucket='wool-exports', Key=key)['Body'].read()

        # Write to Wool Storage
        client.write(file_id, data)

        print(f"Imported {file_id}")
```

---

## Rollback Procedures

### Rolling Back a Failed Upgrade

**If new version has issues**:

```bash
# 1. Check rollout status
kubectl rollout status statefulset/wool-storage -n wool-storage

# 2. Rollback to previous version
kubectl rollout undo statefulset/wool-storage -n wool-storage

# 3. Verify rollback
kubectl rollout status statefulset/wool-storage -n wool-storage

# 4. Check version
for i in {0..2}; do
  kubectl exec wool-storage-$i -n wool-storage -- curl -s localhost:9000/version
done
# Should show previous version
```

### Rollback from Backup

**If data corruption or critical bug**:

```bash
# 1. Stop cluster
kubectl scale statefulset wool-storage --replicas=0 -n wool-storage

# 2. Delete PVCs (data loss!)
kubectl delete pvc -l app=wool-storage -n wool-storage

# 3. Restore from backup
velero restore create wool-rollback \
  --from-backup wool-storage-daily-20251117 \
  --wait

# 4. Verify data
kubectl exec wool-storage-0 -n wool-storage -- curl -s localhost:9000/stats | jq '.files_total'

# 5. Resume traffic
kubectl scale statefulset wool-storage --replicas=3 -n wool-storage
```

---

## Migration Checklist

### Pre-Migration

- [ ] **Backup created** (Velero or manual)
- [ ] **Backup verified** (can restore successfully)
- [ ] **Target cluster deployed** and healthy
- [ ] **Monitoring setup** (Grafana dashboards)
- [ ] **Rollback plan** documented
- [ ] **Communication sent** to users (maintenance window)

### During Migration

- [ ] **Enable read-only mode** on source cluster (optional)
- [ ] **Data copied** to target cluster
- [ ] **Replication complete** (factor = 3.0)
- [ ] **Integrity check passed** on all nodes
- [ ] **Spot-check critical files** verified

### Post-Migration

- [ ] **Traffic switched** to new cluster
- [ ] **Client endpoints updated**
- [ ] **Monitoring active** (no errors)
- [ ] **Old cluster in standby** (1-2 weeks)
- [ ] **Final backup** of old cluster
- [ ] **Decommission old cluster**
- [ ] **Archive backups** for compliance

---

**Author**: Claude Code
**Date**: November 18, 2025
**Status**: Production Ready
**Version**: 1.0
