# Complete System Failure Recovery Playbook

**Created**: 2025-11-16
**Last Updated**: 2025-11-16
**RTO Target**: 30 minutes
**RPO Target**: 24 hours

## Overview

This playbook covers recovery from complete system failure, including total infrastructure loss, hardware failure, or catastrophic software failure requiring full system rebuild.

## Symptoms

- All services unresponsive
- Server unreachable (SSH timeout)
- Kubernetes cluster down
- Docker daemon not running
- Multiple component failures
- Complete infrastructure outage

## Prerequisites

- Access to backup infrastructure (S3, separate region)
- Fresh server or VM available
- Infrastructure as Code (Terraform/Kubernetes manifests)
- Root/admin access
- Network connectivity

## Recovery Steps

### Step 1: Assess Scope of Failure

**Time**: 2-5 minutes

```bash
# Test network connectivity
ping voice.hololoom.ai

# Test SSH access
ssh ops@voice-server.hololoom.ai

# Check cloud provider status
# AWS: https://status.aws.amazon.com/
# GCP: https://status.cloud.google.com/

# Check monitoring
# Grafana, Prometheus, DataDog, etc.
```

**Decision Tree**:
- If single server failure → Proceed to infrastructure rebuild
- If regional failure → Failover to backup region (see multi_region_failover.md)
- If cloud provider outage → Wait or failover to different provider

### Step 2: Provision New Infrastructure

**Time**: 5-10 minutes

#### Option A: Cloud (AWS/GCP)

```bash
# Clone infrastructure repository
git clone https://github.com/hololoom/infrastructure.git
cd infrastructure/terraform

# Initialize Terraform
terraform init

# Provision new server
terraform apply -var="environment=production" -var="region=us-east-1"

# Get new server IP
NEW_SERVER_IP=$(terraform output -raw server_ip)
echo "New server: $NEW_SERVER_IP"
```

#### Option B: Kubernetes

```bash
# Scale up deployment
kubectl scale deployment hololoom-voice-agent --replicas=3

# Or provision new cluster
eksctl create cluster -f cluster-config.yaml
```

#### Option C: Manual VM

```bash
# Create Ubuntu 22.04 VM with:
# - 4 vCPUs
# - 16GB RAM
# - 100GB SSD
# - Docker installed
# - Docker Compose installed
```

### Step 3: Install System Dependencies

**Time**: 5 minutes

```bash
# SSH to new server
ssh root@$NEW_SERVER_IP

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt-get update
apt-get install -y docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

### Step 4: Clone Application Repository

**Time**: 2 minutes

```bash
# Clone HoloLoom repository
cd /opt
git clone https://github.com/hololoom/hololoom.git
cd hololoom

# Checkout production branch
git checkout production

# Verify files
ls -la docker-compose.voice.yml scripts/
```

### Step 5: Download Latest Backup

**Time**: 3-5 minutes

```bash
# Install AWS CLI if needed
apt-get install -y awscli

# Configure AWS credentials
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Format (json)

# List available backups
aws s3 ls s3://hololoom-backups/ | grep hololoom_backup | tail -10

# Download latest backup
LATEST_BACKUP=$(aws s3 ls s3://hololoom-backups/ | grep hololoom_backup | tail -1 | awk '{print $4}')
aws s3 cp "s3://hololoom-backups/$LATEST_BACKUP" /var/backups/hololoom/

# Verify integrity
aws s3 cp "s3://hololoom-backups/$LATEST_BACKUP.sha256" /var/backups/hololoom/
cd /var/backups/hololoom
sha256sum -c "$LATEST_BACKUP.sha256"
# Should output: "OK"
```

### Step 6: Restore from Backup

**Time**: 10-15 minutes

```bash
# Run disaster recovery script
cd /opt/hololoom
bash scripts/disaster_recovery.sh \
  /var/backups/hololoom/$LATEST_BACKUP \
  --force

# Script will:
# 1. Extract backup
# 2. Create Docker volumes
# 3. Restore Neo4j
# 4. Restore Redis
# 5. Restore configurations
# 6. Start all services
# 7. Validate health
```

### Step 7: Update DNS/Load Balancer

**Time**: 2-5 minutes

```bash
# Update DNS record (Route53 example)
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "voice.hololoom.ai",
        "Type": "A",
        "TTL": 60,
        "ResourceRecords": [{"Value": "'$NEW_SERVER_IP'"}]
      }
    }]
  }'

# Or update load balancer target
aws elbv2 register-targets \
  --target-group-arn arn:aws:elasticloadbalancing:... \
  --targets Id=$NEW_SERVER_IP
```

### Step 8: Verify System Functionality

**Time**: 5 minutes

```bash
# Wait for DNS propagation
sleep 60

# Test health endpoints
curl https://voice.hololoom.ai/health | jq '.'

# Test voice query
curl -X POST https://voice.hololoom.ai/voice/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "System recovery test",
    "language": "en"
  }' | jq '.'

# Test AR integration
curl https://voice.hololoom.ai/ar/status | jq '.'

# Check all components
bash scripts/validate_deployment.sh
```

### Step 9: Restore Monitoring

**Time**: 3 minutes

```bash
# Restart Prometheus
docker-compose -f docker-compose.voice.yml up -d prometheus

# Restart Grafana
docker-compose -f docker-compose.voice.yml up -d grafana

# Verify metrics collection
curl http://localhost:9090/-/healthy
curl http://localhost:3000/api/health

# Import Grafana dashboards (if not in backup)
# Dashboards are restored from backup automatically
```

### Step 10: Enable Automated Backups

**Time**: 2 minutes

```bash
# Set up cron job for backups
crontab -e

# Add daily backup at 2 AM
# 0 2 * * * /opt/hololoom/scripts/backup_automation.sh
```

### Step 11: Document Incident

**Time**: 5 minutes

```bash
# Create incident report
cat > /var/log/hololoom/incident_$(date +%Y%m%d_%H%M%S).txt <<EOF
Incident: Complete System Failure
Date: $(date)
Severity: CRITICAL
Duration: [Start time] to [End time] = [Total minutes]

Failure Details:
- Affected Components: All
- Root Cause: [Describe: hardware, software, network, etc.]
- Detection Method: [How discovered]

Recovery Actions:
1. Provisioned new infrastructure: $NEW_SERVER_IP
2. Restored from backup: $LATEST_BACKUP
3. Updated DNS/LB to new server
4. Verified all functionality

RTO Achievement: [Actual minutes] (Target: 30 minutes)
RPO: [Data loss in hours] (Target: 24 hours)

Data Loss: [None / List affected data]
User Impact: [Number of users, duration]

Post-Mortem:
- What went well: [...]
- What can improve: [...]
- Action items: [...]

Follow-up:
- [ ] Root cause analysis
- [ ] Infrastructure hardening
- [ ] Disaster recovery drill
- [ ] Update playbooks
EOF
```

## Post-Recovery Checklist

- [ ] All services running (voice-agent, neo4j, redis, prometheus, grafana)
- [ ] Health endpoints return 200 OK
- [ ] Voice queries work end-to-end
- [ ] AR integration functional
- [ ] Knowledge graph accessible
- [ ] Metrics being collected
- [ ] Dashboards populated
- [ ] DNS/Load balancer updated
- [ ] Monitoring alerts active
- [ ] Automated backups scheduled
- [ ] Incident report completed
- [ ] Stakeholders notified

## RTO/RPO Analysis

**RTO Achieved**:
```
Provision Infrastructure: 5-10 min
Install Dependencies: 5 min
Download Backup: 3-5 min
Restore System: 10-15 min
Update DNS: 2-5 min
Verify System: 5 min
Total: 30-45 minutes
```

**Target**: 30 minutes ✓ (if infrastructure pre-provisioned)

**RPO Achieved**:
- Daily backups: 0-24 hours data loss
- Transaction logs: Potential <1 hour data loss

**Target**: 24 hours ✓

## Optimization Strategies

### To achieve <30 min RTO:

1. **Pre-provision hot standby**:
   ```bash
   # Keep standby server running
   # Replicate data in real-time
   # Instant failover with DNS update
   ```

2. **Use Infrastructure as Code**:
   ```bash
   # Terraform apply in <5 minutes
   # Pre-baked AMI/VM images
   ```

3. **Automate entire recovery**:
   ```bash
   # One-command recovery
   ./scripts/complete_recovery.sh --backup latest --region us-east-1
   ```

### To achieve <1 hour RPO:

1. **Hourly incremental backups**:
   ```bash
   # Cron: 0 * * * *
   ```

2. **Enable Neo4j transaction log shipping**:
   ```yaml
   NEO4J_dbms_tx__log_rotation_retention__policy: 2 days
   ```

3. **Real-time replication to standby**

## Failover to Backup Region

If primary region is completely unavailable:

```bash
# Activate failover manager
python3 scripts/activate_failover.py --region us-west-2

# Update global load balancer
# Route traffic to us-west-2
```

See [multi_region_failover.md](multi_region_failover.md) for details.

## Communication Template

**Email to Stakeholders**:

```
Subject: [RESOLVED] HoloLoom Voice System Outage - [Date]

Team,

We experienced a complete system outage affecting HoloLoom Voice services from [START_TIME] to [END_TIME] ([DURATION] minutes).

Impact:
- All voice queries unavailable
- [NUMBER] users affected
- [FEATURES] temporarily offline

Root Cause:
[BRIEF DESCRIPTION]

Resolution:
- Provisioned new infrastructure
- Restored from latest backup ([BACKUP_TIME])
- Verified all functionality

Data Loss: [NONE / DESCRIPTION]

System Status: FULLY OPERATIONAL

Actions Taken:
1. [...]
2. [...]

Next Steps:
- Root cause analysis (ETA: [DATE])
- Infrastructure hardening
- Disaster recovery drill

Contact ops@hololoom.ai for questions.

Best regards,
HoloLoom Operations Team
```

## Escalation

If recovery exceeds 30 minutes:

1. **Notify**: VP Engineering
2. **Slack**: #incidents-critical
3. **Email**: exec-oncall@hololoom.ai
4. **Consider**: External vendor support (AWS Premium Support, etc.)

## Prevention

1. **Multi-region deployment**:
   - Deploy to 2+ regions
   - Active-active or active-passive

2. **Chaos engineering**:
   - Regular disaster recovery drills
   - Simulate failures monthly

3. **Infrastructure as Code**:
   - Terraform for all infrastructure
   - Version controlled

4. **Monitoring**:
   - Uptime checks (external)
   - PagerDuty integration
   - Auto-scaling

## References

- [Disaster Recovery README](../DISASTER_RECOVERY_README.md)
- [Multi-Region Failover Playbook](multi_region_failover.md)
- [Infrastructure Repository](https://github.com/hololoom/infrastructure)
- [AWS Disaster Recovery Guide](https://docs.aws.amazon.com/whitepapers/latest/disaster-recovery-workloads-on-aws/disaster-recovery-workloads-on-aws.html)

---

**Last Reviewed**: 2025-11-16
**Reviewer**: Agent H - Wave 3 Production Hardening
**Next Review**: 2025-12-16
