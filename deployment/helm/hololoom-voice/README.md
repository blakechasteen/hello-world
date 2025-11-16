# HoloLoom VoiceAgent Helm Chart

**Version**: 1.0.0
**App Version**: 1.0.0
**Date**: November 15, 2025

---

## Overview

Helm chart for deploying the **HoloLoom VoiceAgent** - a production-ready voice interaction system with bidirectional audio, TTS synthesis, and neural decision-making integration.

### Features

- ✅ **Production-Ready**: Zero-downtime deployments, autoscaling, health checks
- ✅ **Highly Available**: 3+ replicas, pod disruption budgets, anti-affinity rules
- ✅ **Observable**: Prometheus metrics, Grafana dashboards, structured logging
- ✅ **Secure**: RBAC, network policies, non-root containers, secret management
- ✅ **Scalable**: Horizontal pod autoscaling (3-20 replicas)
- ✅ **Persistent**: Voice session storage with PVCs

---

## Prerequisites

- Kubernetes 1.21+
- Helm 3.8+
- Persistent volume provisioner (for session storage)
- Optional: Prometheus Operator (for ServiceMonitor)
- Optional: Cert-Manager (for TLS certificates)

---

## Quick Start

### 1. Add Helm Repository

```bash
# If using a Helm repository
helm repo add hololoom https://charts.hololoom.ai
helm repo update
```

### 2. Create Namespace

```bash
kubectl create namespace hololoom-voice
```

### 3. Create Secrets

```bash
# Create OpenAI API key secret
kubectl create secret generic voice-agent-secrets \
  --from-literal=OPENAI_API_KEY='sk-your-openai-key' \
  --from-literal=NEO4J_PASSWORD='your-neo4j-password' \
  -n hololoom-voice
```

### 4. Install Chart

```bash
# Basic installation
helm install hololoom-voice hololoom/hololoom-voice \
  --namespace hololoom-voice

# Or from local chart
helm install hololoom-voice ./deployment/helm/hololoom-voice \
  --namespace hololoom-voice
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods -n hololoom-voice

# Check services
kubectl get svc -n hololoom-voice

# Check logs
kubectl logs -n hololoom-voice -l app=voice-agent -f
```

---

## Configuration

### Override Values

Create a custom `values.yaml`:

```yaml
# custom-values.yaml
voiceAgent:
  replicaCount: 5

  resources:
    requests:
      memory: "1Gi"
      cpu: "1000m"
    limits:
      memory: "4Gi"
      cpu: "4000m"

  autoscaling:
    minReplicas: 5
    maxReplicas: 50

  env:
    - name: HOLOLOOM_MODE
      value: "RESEARCH"  # BARE, FAST, FUSED, RESEARCH
    - name: LOG_LEVEL
      value: "DEBUG"

persistence:
  voiceSessions:
    size: 100Gi
    storageClass: "fast-ssd"
```

Install with custom values:

```bash
helm install hololoom-voice ./deployment/helm/hololoom-voice \
  --namespace hololoom-voice \
  --values custom-values.yaml
```

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `voiceAgent.replicaCount` | Number of replicas | `3` |
| `voiceAgent.image.tag` | Image tag | `1.0.0` |
| `voiceAgent.autoscaling.enabled` | Enable HPA | `true` |
| `voiceAgent.autoscaling.minReplicas` | Min replicas | `3` |
| `voiceAgent.autoscaling.maxReplicas` | Max replicas | `20` |
| `voiceAgent.resources.requests.memory` | Memory request | `512Mi` |
| `voiceAgent.resources.limits.memory` | Memory limit | `2Gi` |
| `persistence.enabled` | Enable persistent storage | `true` |
| `persistence.voiceSessions.size` | Session storage size | `50Gi` |
| `monitoring.prometheus.enabled` | Enable Prometheus | `true` |
| `networkPolicy.enabled` | Enable network policies | `true` |

---

## Environment Variables

### Core Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `HOLOLOOM_MODE` | Processing mode | `FUSED` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENABLE_METRICS` | Enable Prometheus metrics | `true` |

### Audio Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SAMPLE_RATE` | Audio sample rate (Hz) | `16000` |
| `CHUNK_DURATION` | Audio chunk duration (seconds) | `1.0` |
| `MAX_QUEUE_SIZE` | Max audio queue size | `100` |

### TTS Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TTS_PROVIDER` | TTS provider | `openai` |
| `TTS_VOICE` | Voice ID | `nova` |

### Turn-Taking Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `TURN_MODE` | Turn-taking mode | `hybrid` |
| `SILENCE_THRESHOLD` | Silence threshold (seconds) | `1.5` |

---

## Scaling

### Manual Scaling

```bash
# Scale to 10 replicas
kubectl scale deployment hololoom-voice-hololoom-voice \
  --replicas=10 \
  -n hololoom-voice
```

### Horizontal Pod Autoscaling

The chart includes HPA by default:

```yaml
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

View HPA status:

```bash
kubectl get hpa -n hololoom-voice
```

---

## Monitoring

### Prometheus Metrics

The chart exposes metrics at `/metrics`:

- `audio_chunks_processed_total` - Total audio chunks
- `extraction_latency_seconds` - Event extraction latency
- `transcription_latency_seconds` - Transcription latency
- `synthesis_latency_seconds` - TTS synthesis latency
- `active_sessions` - Active voice sessions
- `extraction_failures_total` - Extraction failures
- `audio_chunks_dropped_total` - Dropped chunks

### ServiceMonitor

If Prometheus Operator is installed:

```yaml
monitoring:
  prometheus:
    servicemonitor:
      enabled: true
      interval: 30s
```

### Grafana Dashboards

Import dashboards from `deployment/grafana/dashboards/`.

---

## Security

### Network Policies

The chart includes network policies to restrict traffic:

```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: hololoom-voice
  egress:
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 7687  # Neo4j
        - protocol: TCP
          port: 6333  # Qdrant
        - protocol: TCP
          port: 443   # HTTPS
```

### RBAC

ServiceAccount with minimal permissions:

```yaml
rbac:
  create: true

serviceAccount:
  create: true
  name: voice-agent-sa
```

### Security Context

Runs as non-root user:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop:
      - ALL
```

---

## Upgrading

### Zero-Downtime Upgrades

```bash
# Upgrade with new values
helm upgrade hololoom-voice ./deployment/helm/hololoom-voice \
  --namespace hololoom-voice \
  --values custom-values.yaml

# Rollback if needed
helm rollback hololoom-voice -n hololoom-voice
```

### Rolling Update Strategy

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1
    maxUnavailable: 0  # Zero downtime
```

---

## Uninstalling

```bash
# Uninstall chart
helm uninstall hololoom-voice -n hololoom-voice

# Delete namespace (caution: deletes all resources)
kubectl delete namespace hololoom-voice
```

---

## Troubleshooting

### Check Pod Status

```bash
kubectl get pods -n hololoom-voice
kubectl describe pod <pod-name> -n hololoom-voice
```

### View Logs

```bash
# All pods
kubectl logs -n hololoom-voice -l app=voice-agent --tail=100

# Specific pod
kubectl logs -n hololoom-voice <pod-name> -f
```

### Check Events

```bash
kubectl get events -n hololoom-voice --sort-by='.lastTimestamp'
```

### Common Issues

**1. Pods not starting**

Check init containers:
```bash
kubectl logs -n hololoom-voice <pod-name> -c wait-for-neo4j
kubectl logs -n hololoom-voice <pod-name> -c wait-for-qdrant
```

**2. Missing secrets**

Verify secrets exist:
```bash
kubectl get secrets -n hololoom-voice
kubectl describe secret voice-agent-secrets -n hololoom-voice
```

**3. PVC not binding**

Check PVC status:
```bash
kubectl get pvc -n hololoom-voice
kubectl describe pvc hololoom-voice-hololoom-voice-sessions -n hololoom-voice
```

**4. HPA not scaling**

Check metrics server:
```bash
kubectl top nodes
kubectl top pods -n hololoom-voice
```

---

## Examples

### Minimal Installation

```bash
helm install hololoom-voice ./deployment/helm/hololoom-voice \
  --namespace hololoom-voice \
  --set voiceAgent.replicaCount=1 \
  --set voiceAgent.autoscaling.enabled=false \
  --set persistence.enabled=false
```

### Production Installation

```bash
helm install hololoom-voice ./deployment/helm/hololoom-voice \
  --namespace hololoom-voice \
  --set voiceAgent.replicaCount=5 \
  --set voiceAgent.autoscaling.minReplicas=5 \
  --set voiceAgent.autoscaling.maxReplicas=50 \
  --set persistence.voiceSessions.size=100Gi \
  --set persistence.voiceSessions.storageClass=fast-ssd \
  --set monitoring.prometheus.enabled=true \
  --set networkPolicy.enabled=true
```

### With Ingress

```bash
helm install hololoom-voice ./deployment/helm/hololoom-voice \
  --namespace hololoom-voice \
  --set voiceAgent.ingress.enabled=true \
  --set voiceAgent.ingress.className=nginx \
  --set voiceAgent.ingress.hosts[0].host=voice.example.com \
  --set voiceAgent.ingress.tls[0].secretName=voice-tls \
  --set voiceAgent.ingress.tls[0].hosts[0]=voice.example.com
```

---

## Advanced Configuration

### Custom Environment Variables

```yaml
voiceAgent:
  env:
    - name: CUSTOM_VAR
      value: "custom_value"
    - name: SECRET_VAR
      valueFrom:
        secretKeyRef:
          name: my-secret
          key: my-key
```

### Custom Affinity Rules

```yaml
voiceAgent:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: node-type
                operator: In
                values:
                  - gpu
```

### Custom Tolerations

```yaml
voiceAgent:
  tolerations:
    - key: "gpu"
      operator: "Equal"
      value: "true"
      effect: "NoSchedule"
```

---

## Support

- **Documentation**: [PRODUCTION_DEPLOYMENT_GUIDE.md](../../PRODUCTION_DEPLOYMENT_GUIDE.md)
- **Issues**: GitHub Issues
- **Community**: Discord / Slack

---

## License

Same as HoloLoom project license.

---

**Version**: 1.0.0
**Status**: ✅ Production Ready
**Last Updated**: November 15, 2025
