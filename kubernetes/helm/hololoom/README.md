# HoloLoom Helm Chart

Official Helm chart for deploying HoloLoom neural decision-making system to Kubernetes.

## Introduction

This chart bootstraps a complete HoloLoom deployment with:
- API Gateway (FastAPI)
- Worker pool (Celery)
- Neo4j knowledge graph
- Qdrant vector database
- Redis cache
- RabbitMQ message queue
- Prometheus + Grafana monitoring

## Prerequisites

- Kubernetes 1.25+
- Helm 3.10+
- PV provisioner support in the underlying infrastructure (for persistent data)

## Installing the Chart

### From source

```bash
# Clone repository
git clone https://github.com/hololoom/hololoom.git
cd hololoom

# Install chart
helm install hololoom kubernetes/helm/hololoom \
  --namespace hololoom \
  --create-namespace \
  --values kubernetes/helm/hololoom/values.yaml
```

### Custom values

```bash
# Create custom values file
cat > my-values.yaml <<EOF
global:
  environment: production

api:
  replicaCount: 5

workers:
  autoscaling:
    enabled: true
    maxReplicas: 20
EOF

# Install with custom values
helm install hololoom kubernetes/helm/hololoom \
  --namespace hololoom \
  --create-namespace \
  --values my-values.yaml
```

## Upgrading the Chart

```bash
helm upgrade hololoom kubernetes/helm/hololoom \
  --namespace hololoom \
  --values my-values.yaml
```

## Uninstalling the Chart

```bash
helm uninstall hololoom --namespace hololoom

# Also delete PVCs if needed
kubectl delete pvc -n hololoom -l app.kubernetes.io/name=hololoom
```

## Configuration

The following table lists the configurable parameters of the HoloLoom chart and their default values.

### Global Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.namespace` | Kubernetes namespace | `hololoom` |
| `global.imageRegistry` | Container image registry | `docker.io` |
| `global.imagePullSecrets` | Image pull secrets | `[]` |
| `global.storageClass` | Storage class for PVCs | `standard` |
| `global.environment` | Environment name | `production` |

### API Gateway Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `api.enabled` | Enable API gateway | `true` |
| `api.replicaCount` | Number of API replicas | `3` |
| `api.image.repository` | API image repository | `hololoom/api` |
| `api.image.tag` | API image tag | `1.0.0` |
| `api.resources.requests.memory` | Memory request | `512Mi` |
| `api.resources.requests.cpu` | CPU request | `500m` |
| `api.resources.limits.memory` | Memory limit | `2Gi` |
| `api.resources.limits.cpu` | CPU limit | `2000m` |
| `api.autoscaling.enabled` | Enable HPA | `false` |
| `api.autoscaling.minReplicas` | Minimum replicas | `3` |
| `api.autoscaling.maxReplicas` | Maximum replicas | `10` |

### Worker Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `workers.enabled` | Enable workers | `true` |
| `workers.replicaCount` | Number of worker replicas | `2` |
| `workers.image.repository` | Worker image repository | `hololoom/worker` |
| `workers.image.tag` | Worker image tag | `1.0.0` |
| `workers.resources.requests.memory` | Memory request | `1Gi` |
| `workers.resources.requests.cpu` | CPU request | `1000m` |
| `workers.autoscaling.enabled` | Enable HPA | `true` |
| `workers.autoscaling.minReplicas` | Minimum replicas | `2` |
| `workers.autoscaling.maxReplicas` | Maximum replicas | `10` |
| `workers.concurrency` | Celery concurrency | `4` |

### Neo4j Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `neo4j.enabled` | Enable Neo4j | `true` |
| `neo4j.replicaCount` | Number of Neo4j nodes | `3` |
| `neo4j.image.repository` | Neo4j image | `neo4j` |
| `neo4j.image.tag` | Neo4j version | `5.13.0-enterprise` |
| `neo4j.persistence.enabled` | Enable persistence | `true` |
| `neo4j.persistence.size` | PVC size | `50Gi` |
| `neo4j.resources.requests.memory` | Memory request | `2Gi` |
| `neo4j.resources.limits.memory` | Memory limit | `8Gi` |

### Qdrant Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `qdrant.enabled` | Enable Qdrant | `true` |
| `qdrant.replicaCount` | Number of Qdrant nodes | `3` |
| `qdrant.image.repository` | Qdrant image | `qdrant/qdrant` |
| `qdrant.image.tag` | Qdrant version | `v1.7.0` |
| `qdrant.persistence.enabled` | Enable persistence | `true` |
| `qdrant.persistence.size` | PVC size | `30Gi` |

### Redis Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `redis.enabled` | Enable Redis | `true` |
| `redis.architecture` | Architecture mode | `replication` |
| `redis.master.replicaCount` | Master replicas | `1` |
| `redis.replica.replicaCount` | Replica count | `2` |
| `redis.persistence.enabled` | Enable persistence | `true` |
| `redis.persistence.size` | PVC size | `8Gi` |

### RabbitMQ Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rabbitmq.enabled` | Enable RabbitMQ | `true` |
| `rabbitmq.replicaCount` | Number of RabbitMQ nodes | `3` |
| `rabbitmq.image.repository` | RabbitMQ image | `rabbitmq` |
| `rabbitmq.image.tag` | RabbitMQ version | `3.12-management-alpine` |
| `rabbitmq.persistence.enabled` | Enable persistence | `true` |
| `rabbitmq.persistence.size` | PVC size | `10Gi` |

### Monitoring Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `monitoring.enabled` | Enable monitoring | `true` |
| `monitoring.prometheus.enabled` | Enable Prometheus | `true` |
| `monitoring.prometheus.persistence.size` | Prometheus PVC size | `20Gi` |
| `monitoring.grafana.enabled` | Enable Grafana | `true` |
| `monitoring.grafana.adminUser` | Grafana admin user | `admin` |
| `monitoring.alertmanager.enabled` | Enable AlertManager | `true` |

### Ingress Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ingress.enabled` | Enable ingress | `true` |
| `ingress.className` | Ingress class | `nginx` |
| `ingress.hosts[0].host` | API host | `api.hololoom.io` |
| `ingress.tls[0].secretName` | TLS secret | `hololoom-tls` |

### Secrets Parameters

**IMPORTANT**: Change all default passwords before production deployment!

| Parameter | Description | Default |
|-----------|-------------|---------|
| `secrets.neo4j.password` | Neo4j password | `change-me-in-production` |
| `secrets.redis.password` | Redis password | `""` |
| `secrets.rabbitmq.username` | RabbitMQ username | `hololoom` |
| `secrets.rabbitmq.password` | RabbitMQ password | `change-me-in-production` |
| `secrets.grafana.adminPassword` | Grafana password | `change-me-in-production` |
| `secrets.api.secretKey` | API secret key | `change-me-in-production` |
| `secrets.api.jwtSecret` | JWT secret | `change-me-in-production` |

## Examples

### Production Deployment with Custom Settings

```yaml
# production-values.yaml
global:
  namespace: hololoom-prod
  environment: production
  storageClass: fast-ssd

api:
  replicaCount: 5
  resources:
    requests:
      memory: "1Gi"
      cpu: "1000m"
    limits:
      memory: "4Gi"
      cpu: "4000m"
  autoscaling:
    enabled: true
    maxReplicas: 15

workers:
  replicaCount: 5
  autoscaling:
    enabled: true
    minReplicas: 5
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70

neo4j:
  replicaCount: 5
  persistence:
    size: 100Gi
  resources:
    requests:
      memory: "4Gi"
    limits:
      memory: "16Gi"

ingress:
  enabled: true
  hosts:
    - host: api.production.hololoom.io
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: production-tls
      hosts:
        - api.production.hololoom.io

secrets:
  neo4j:
    password: "<secure-password>"
  rabbitmq:
    password: "<secure-password>"
  grafana:
    adminPassword: "<secure-password>"
  api:
    secretKey: "<secure-random-key>"
    jwtSecret: "<secure-random-key>"
```

Deploy:

```bash
helm install hololoom-prod kubernetes/helm/hololoom \
  --namespace hololoom-prod \
  --create-namespace \
  --values production-values.yaml
```

### Development Deployment (Minimal Resources)

```yaml
# dev-values.yaml
global:
  namespace: hololoom-dev
  environment: development

api:
  replicaCount: 1
  resources:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "1000m"

workers:
  replicaCount: 1
  autoscaling:
    enabled: false

neo4j:
  replicaCount: 1
  persistence:
    size: 10Gi

qdrant:
  replicaCount: 1
  persistence:
    size: 5Gi

redis:
  architecture: standalone
  replica:
    replicaCount: 0

rabbitmq:
  replicaCount: 1

monitoring:
  enabled: false
```

### Disable Components

```yaml
# minimal-values.yaml
# Deploy only API and workers (use external databases)

neo4j:
  enabled: false

qdrant:
  enabled: false

redis:
  enabled: false

rabbitmq:
  enabled: false

monitoring:
  enabled: false

configMap:
  database:
    NEO4J_URI: "neo4j://external-neo4j:7687"
    QDRANT_URL: "http://external-qdrant:6333"
    REDIS_URL: "redis://external-redis:6379/0"
  queue:
    RABBITMQ_URL: "amqp://external-rabbitmq:5672"
```

## Troubleshooting

### Check Release Status

```bash
helm status hololoom -n hololoom
helm get all hololoom -n hololoom
```

### View Rendered Templates

```bash
helm template hololoom kubernetes/helm/hololoom \
  --values my-values.yaml \
  --debug
```

### Common Issues

1. **Image Pull Errors**: Ensure image registry is accessible and credentials are configured
2. **PVC Binding**: Check storage class exists and has available capacity
3. **Insufficient Resources**: Nodes need enough CPU/memory for requested resources
4. **DNS Resolution**: Verify CoreDNS is running and services are created

### Get Help

Run health check:

```bash
./kubernetes/scripts/health-check.sh -n hololoom --verbose
```

View logs:

```bash
kubectl logs -n hololoom -l app.kubernetes.io/name=hololoom --tail=100
```

## Development

### Linting

```bash
helm lint kubernetes/helm/hololoom
```

### Testing

```bash
# Dry-run
helm install hololoom-test kubernetes/helm/hololoom \
  --namespace hololoom-test \
  --dry-run --debug

# Test with helm test (after installation)
helm test hololoom -n hololoom
```

## License

MIT

## Maintainers

- HoloLoom Team (team@hololoom.io)

## Links

- [GitHub Repository](https://github.com/hololoom/hololoom)
- [Documentation](https://docs.hololoom.io)
- [Full Kubernetes Guide](../../../KUBERNETES.md)
