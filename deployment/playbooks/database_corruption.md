# Database Corruption Recovery Playbook

**Created**: 2025-11-16
**Last Updated**: 2025-11-16
**RTO Target**: 30 minutes
**RPO Target**: 24 hours

## Overview

This playbook covers recovery from Neo4j knowledge graph corruption, including data inconsistencies, index corruption, and database file corruption.

## Symptoms

- Neo4j fails to start with error logs
- Database consistency check failures
- Query failures with "database corrupted" errors
- Transaction log replay errors
- Index corruption warnings
- Data inconsistency errors (missing relationships, orphaned nodes)

## Prerequisites

- Latest backup available (`/var/backups/hololoom` or S3)
- Database administrator access
- Docker and docker-compose installed
- Sufficient disk space for restoration

## Recovery Steps

### Step 1: Assess Damage

**Time**: 2-5 minutes

```bash
# Check Neo4j logs for errors
docker logs hololoom-neo4j --tail 100

# Run consistency check (if Neo4j can start)
docker exec hololoom-neo4j neo4j-admin check-consistency \
  --database=neo4j \
  --report-dir=/var/lib/neo4j/reports

# View consistency report
docker exec hololoom-neo4j cat /var/lib/neo4j/reports/consistency_check.report
```

**Decision Point**:
- If corruption is minor (< 1% of data): Attempt repair
- If corruption is major (> 1% of data): Restore from backup

### Step 2: Attempt Repair (Minor Corruption Only)

**Time**: 5-10 minutes

```bash
# Stop Neo4j
docker-compose -f docker-compose.voice.yml down neo4j

# Rebuild indexes (often fixes corruption)
docker run --rm \
  -v neo4j_data:/data \
  neo4j:5 \
  neo4j-admin database rebuild-indexes neo4j

# Restart Neo4j
docker-compose -f docker-compose.voice.yml up -d neo4j

# Verify repair
docker exec hololoom-neo4j cypher-shell "MATCH (n) RETURN count(n) LIMIT 1"
```

If repair successful, **STOP HERE**. Otherwise, proceed to backup restoration.

### Step 3: Stop All Services

**Time**: 1 minute

```bash
# Stop all HoloLoom services
docker-compose -f docker-compose.voice.yml down

# Verify all containers stopped
docker ps | grep hololoom
# Should show no results
```

### Step 4: Identify Latest Backup

**Time**: 1 minute

```bash
# List local backups
ls -lth /var/backups/hololoom/hololoom_backup_*.tar.gz | head -5

# Or list S3 backups
aws s3 ls s3://hololoom-backups/ | grep hololoom_backup | tail -5

# Download latest backup from S3 if needed
LATEST_BACKUP=$(aws s3 ls s3://hololoom-backups/ | grep hololoom_backup | tail -1 | awk '{print $4}')
aws s3 cp "s3://hololoom-backups/$LATEST_BACKUP" /tmp/
```

### Step 5: Restore from Backup

**Time**: 10-15 minutes

```bash
# Run disaster recovery script
cd /home/user/hello-world
bash scripts/disaster_recovery.sh /var/backups/hololoom/hololoom_backup_YYYYMMDD_HHMMSS.tar.gz

# Script will:
# 1. Verify backup integrity (checksum)
# 2. Stop services
# 3. Restore Neo4j database
# 4. Restore Redis cache
# 5. Restore configurations
# 6. Restart services
# 7. Validate health
```

**Note**: The script prompts for confirmation. Type `yes` to proceed.

### Step 6: Verify Database Integrity

**Time**: 3-5 minutes

```bash
# Run consistency check
docker exec hololoom-neo4j neo4j-admin check-consistency \
  --database=neo4j \
  --report-dir=/var/lib/neo4j/reports

# Should show "Database is consistent"

# Test basic queries
docker exec hololoom-neo4j cypher-shell "
  MATCH (n) RETURN count(n) AS node_count;
  MATCH ()-[r]->() RETURN count(r) AS relationship_count;
"

# Test knowledge graph queries
docker exec hololoom-neo4j cypher-shell "
  MATCH (e:Entity)-[r:RELATES_TO]->(e2:Entity)
  RETURN e.name, type(r), e2.name
  LIMIT 10;
"
```

### Step 7: Verify Application Functionality

**Time**: 5 minutes

```bash
# Check health endpoints
curl http://localhost:8000/health | jq '.'

# Test voice query
curl -X POST http://localhost:8000/voice/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Test query",
    "language": "en",
    "personality": "default"
  }' | jq '.'

# Check Grafana dashboards
open http://localhost:3000
# Verify data is displaying correctly
```

### Step 8: Document Recovery

**Time**: 2 minutes

```bash
# Create incident report
cat > /var/log/hololoom/incident_$(date +%Y%m%d_%H%M%S).txt <<EOF
Incident: Neo4j Database Corruption
Date: $(date)
Symptoms: [Describe symptoms observed]
Root Cause: [If known]
Recovery Method: Restore from backup
Backup Used: [Backup filename]
Downtime: [Start time] to [End time] = [Duration]
Data Loss: [Describe if any]
Actions Taken: [List steps performed]
Preventive Measures: [List any improvements]
EOF
```

## Post-Recovery Checklist

- [ ] Neo4j consistency check passes
- [ ] All health endpoints return 200 OK
- [ ] Voice queries work correctly
- [ ] Grafana dashboards show data
- [ ] Knowledge graph queries return results
- [ ] No errors in application logs
- [ ] Metrics are being collected
- [ ] Incident report completed

## Data Loss Assessment

**RPO**: 24 hours (daily backups)

If last backup is older than RPO:
1. Check if transaction logs are available
2. Consider manual data recovery from logs
3. Document data loss extent
4. Notify stakeholders

## Common Errors

### Error: "Database dump not found"

**Solution**: Use data directory backup instead
```bash
# In disaster_recovery.sh, this is handled automatically
# Script will copy neo4j_data directory if dump not found
```

### Error: "Disk space full"

**Solution**: Clear old backups and logs
```bash
# Remove old backups
find /var/backups/hololoom -name "hololoom_backup_*.tar.gz" -mtime +30 -delete

# Remove old logs
docker exec hololoom-neo4j bash -c "find /logs -name '*.log.*' -mtime +7 -delete"
```

### Error: "Neo4j won't start after restore"

**Solution**: Check permissions
```bash
# Fix Neo4j data permissions
docker run --rm -v neo4j_data:/data alpine chmod -R 777 /data

# Restart Neo4j
docker-compose -f docker-compose.voice.yml up -d neo4j
```

## Escalation

If recovery fails after following this playbook:

1. **Contact**: Database Team Lead
2. **Email**: db-oncall@hololoom.ai
3. **Slack**: #incidents-critical
4. **Phone**: On-call rotation (see PagerDuty)

Provide:
- Incident report
- Neo4j logs (`docker logs hololoom-neo4j > neo4j_logs.txt`)
- Consistency check report
- Recovery attempt log

## Prevention

To prevent future corruption:

1. **Enable daily automated backups**:
   ```bash
   # Add to crontab
   0 2 * * * /home/user/hello-world/scripts/backup_automation.sh
   ```

2. **Enable Neo4j transaction logs**:
   ```yaml
   # In docker-compose.voice.yml
   environment:
     - NEO4J_dbms_tx__log_rotation_retention__policy=7 days
   ```

3. **Monitor disk space**:
   - Set up Prometheus alert for disk usage > 80%
   - Auto-cleanup old logs

4. **Regular consistency checks**:
   ```bash
   # Weekly cron job
   0 3 * * 0 docker exec hololoom-neo4j neo4j-admin check-consistency --database=neo4j
   ```

## References

- [Neo4j Backup Documentation](https://neo4j.com/docs/operations-manual/current/backup-restore/)
- [Neo4j Consistency Checker](https://neo4j.com/docs/operations-manual/current/tools/consistency-checker/)
- [HoloLoom Disaster Recovery README](../DISASTER_RECOVERY_README.md)

---

**Last Reviewed**: 2025-11-16
**Reviewer**: Agent H - Wave 3 Production Hardening
**Next Review**: 2025-12-16
