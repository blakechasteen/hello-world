# Kubernetes Commands Reference
## EdWIN AI Tutor - Production Operations

**Quick Reference**: Common kubectl commands for EdWIN deployment

---

## 📋 Deployment Commands

### Initial Deployment
```bash
# Deploy all components
./scripts/deploy.sh staging

# Or manually:
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.production.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/
```

### Update Deployment
```bash
# Update specific deployment
kubectl set image deployment/edwin-api \
  edwin-api=ghcr.io/your-org/edwin-api:v1.1.0 \
  -n edwin

# Or apply changes
kubectl apply -f k8s/api-deployment.yaml

# Rollout restart (without image change)
kubectl rollout restart deployment/edwin-api -n edwin
```

### Rollback
```bash
# Automatic rollback
./scripts/rollback.sh production

# Or manually:
kubectl rollout undo deployment/edwin-api -n edwin
kubectl rollout undo deployment/edwin-dashboard -n edwin
kubectl rollout undo deployment/edwin-mobile -n edwin

# Rollback to specific revision
kubectl rollout history deployment/edwin-api -n edwin
kubectl rollout undo deployment/edwin-api --to-revision=3 -n edwin
```

---

## 🔍 Viewing Resources

### Pods
```bash
# List all pods
kubectl get pods -n edwin

# Watch pods (auto-refresh)
kubectl get pods -n edwin -w

# Detailed pod info
kubectl describe pod edwin-api-xxx -n edwin

# Pod status with more details
kubectl get pods -n edwin -o wide

# Filter by label
kubectl get pods -n edwin -l app=edwin-api
```

### Deployments
```bash
# List deployments
kubectl get deployments -n edwin

# Deployment status
kubectl rollout status deployment/edwin-api -n edwin

# Deployment history
kubectl rollout history deployment/edwin-api -n edwin

# Describe deployment
kubectl describe deployment edwin-api -n edwin
```

### Services
```bash
# List services
kubectl get svc -n edwin

# Service details
kubectl describe svc edwin-api -n edwin

# Service endpoints
kubectl get endpoints -n edwin
```

### StatefulSets
```bash
# List statefulsets
kubectl get statefulsets -n edwin

# StatefulSet status
kubectl describe statefulset neo4j -n edwin
```

### All Resources
```bash
# Everything in namespace
kubectl get all -n edwin

# Specific resource types
kubectl get pods,svc,deployments -n edwin
```

---

## 📊 Logs & Debugging

### View Logs
```bash
# Tail logs (follow)
kubectl logs -f deployment/edwin-api -n edwin

# Last 100 lines
kubectl logs --tail=100 deployment/edwin-api -n edwin

# Logs from specific pod
kubectl logs edwin-api-xxx -n edwin

# Logs from all pods in deployment
kubectl logs -l app=edwin-api -n edwin --all-containers

# Previous pod logs (if crashed)
kubectl logs edwin-api-xxx -n edwin --previous

# Logs with timestamps
kubectl logs -f deployment/edwin-api -n edwin --timestamps
```

### Execute Commands in Pod
```bash
# Interactive shell
kubectl exec -it edwin-api-xxx -n edwin -- /bin/bash

# Single command
kubectl exec edwin-api-xxx -n edwin -- env

# Test database connection
kubectl exec edwin-api-xxx -n edwin -- \
  curl -s http://neo4j:7474

# Check Python packages
kubectl exec edwin-api-xxx -n edwin -- pip list
```

### Debug Networking
```bash
# Port forward to local
kubectl port-forward svc/edwin-api 8000:8000 -n edwin

# Port forward specific pod
kubectl port-forward edwin-api-xxx 8000:8000 -n edwin

# Test service connectivity
kubectl run curl-test --rm -i --restart=Never \
  --image=curlimages/curl:latest \
  -n edwin \
  -- curl -s http://edwin-api:8000/health
```

---

## ⚙️ Scaling

### Manual Scaling
```bash
# Scale deployment
kubectl scale deployment/edwin-api --replicas=5 -n edwin

# Scale multiple deployments
kubectl scale deployment edwin-api edwin-dashboard \
  --replicas=3 -n edwin
```

### Autoscaling (HPA)
```bash
# View HPA status
kubectl get hpa -n edwin

# Describe HPA
kubectl describe hpa edwin-api-hpa -n edwin

# Edit HPA
kubectl edit hpa edwin-api-hpa -n edwin

# Manual autoscale (if HPA not created)
kubectl autoscale deployment edwin-api \
  --min=3 --max=10 \
  --cpu-percent=70 \
  -n edwin
```

---

## 🔐 Secrets & ConfigMaps

### Secrets
```bash
# List secrets
kubectl get secrets -n edwin

# View secret (base64 encoded)
kubectl get secret edwin-secrets -n edwin -o yaml

# Decode secret value
kubectl get secret edwin-secrets -n edwin \
  -o jsonpath='{.data.JWT_SECRET_KEY}' | base64 -d

# Create secret
kubectl create secret generic edwin-secrets -n edwin \
  --from-literal=NEO4J_PASSWORD=password \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 32)

# Update secret (delete + recreate)
kubectl delete secret edwin-secrets -n edwin
kubectl create secret generic edwin-secrets -n edwin \
  --from-file=secrets.yaml
```

### ConfigMaps
```bash
# List configmaps
kubectl get configmaps -n edwin

# View configmap
kubectl get configmap edwin-config -n edwin -o yaml

# Create configmap from file
kubectl create configmap edwin-config -n edwin \
  --from-file=config/production.yaml

# Update configmap
kubectl create configmap edwin-config -n edwin \
  --from-file=config/production.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
```

---

## 📈 Monitoring & Health

### Health Checks
```bash
# Check pod health
kubectl get pods -n edwin

# Detailed health status
kubectl describe pod edwin-api-xxx -n edwin | grep -A 5 Conditions

# Run health check script
./scripts/health_check.sh

# Manual health check
kubectl run health-check --rm -i --restart=Never \
  --image=curlimages/curl:latest \
  -n edwin \
  -- curl -s http://edwin-api:8000/health
```

### Resource Usage
```bash
# Node usage
kubectl top nodes

# Pod usage
kubectl top pods -n edwin

# Deployment resource usage
kubectl top pods -n edwin -l app=edwin-api

# Detailed resource info
kubectl describe nodes | grep -A 5 "Allocated resources"
```

### Events
```bash
# Recent events
kubectl get events -n edwin --sort-by='.lastTimestamp'

# Watch events
kubectl get events -n edwin --watch

# Events for specific resource
kubectl describe pod edwin-api-xxx -n edwin | grep Events -A 20
```

---

## 💾 Persistent Storage

### Persistent Volumes
```bash
# List PVs
kubectl get pv

# List PVCs
kubectl get pvc -n edwin

# Describe PVC
kubectl describe pvc neo4j-data-neo4j-0 -n edwin

# View PV used by PVC
kubectl get pvc neo4j-data-neo4j-0 -n edwin \
  -o jsonpath='{.spec.volumeName}'
```

### Resize PVC
```bash
# Edit PVC size
kubectl edit pvc neo4j-data-neo4j-0 -n edwin

# Or patch
kubectl patch pvc neo4j-data-neo4j-0 -n edwin \
  -p '{"spec":{"resources":{"requests":{"storage":"20Gi"}}}}'
```

---

## 🌐 Ingress & Networking

### Ingress
```bash
# List ingress
kubectl get ingress -n edwin

# Ingress details
kubectl describe ingress edwin-ingress -n edwin

# Edit ingress
kubectl edit ingress edwin-ingress -n edwin

# Test ingress
curl -H "Host: api.edwin.edu" http://<ingress-ip>/health
```

### Network Policies
```bash
# List network policies
kubectl get networkpolicies -n edwin

# Describe policy
kubectl describe networkpolicy allow-api -n edwin
```

---

## 🔄 Updates & Maintenance

### Rolling Update
```bash
# Start rollout
kubectl set image deployment/edwin-api \
  edwin-api=ghcr.io/your-org/edwin-api:v1.2.0 \
  -n edwin

# Watch rollout
kubectl rollout status deployment/edwin-api -n edwin

# Pause rollout
kubectl rollout pause deployment/edwin-api -n edwin

# Resume rollout
kubectl rollout resume deployment/edwin-api -n edwin

# Rollout history
kubectl rollout history deployment/edwin-api -n edwin
```

### Restart Pods
```bash
# Graceful restart (rolling)
kubectl rollout restart deployment/edwin-api -n edwin

# Delete specific pod (recreated automatically)
kubectl delete pod edwin-api-xxx -n edwin

# Delete all pods in deployment (recreated)
kubectl delete pods -n edwin -l app=edwin-api
```

### Drain Node
```bash
# Drain node for maintenance
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data

# Mark node as unschedulable
kubectl cordon <node-name>

# Mark node as schedulable
kubectl uncordon <node-name>
```

---

## 🧹 Cleanup

### Delete Resources
```bash
# Delete specific deployment
kubectl delete deployment edwin-api -n edwin

# Delete all deployments
kubectl delete deployments --all -n edwin

# Delete namespace (deletes everything in it)
kubectl delete namespace edwin

# Delete by label
kubectl delete pods -n edwin -l app=edwin-api
```

### Force Delete
```bash
# Force delete stuck pod
kubectl delete pod edwin-api-xxx -n edwin --force --grace-period=0

# Force delete namespace
kubectl delete namespace edwin --force --grace-period=0
```

---

## 🔧 Troubleshooting

### Pod Not Starting
```bash
# Check pod events
kubectl describe pod edwin-api-xxx -n edwin

# Check logs
kubectl logs edwin-api-xxx -n edwin

# Check previous logs (if crashed)
kubectl logs edwin-api-xxx -n edwin --previous

# Check init containers
kubectl logs edwin-api-xxx -n edwin -c init-container-name
```

### Image Pull Errors
```bash
# Check image pull secret
kubectl get secrets -n edwin

# Describe pod for pull errors
kubectl describe pod edwin-api-xxx -n edwin | grep -A 5 "Failed to pull"

# Test image pull manually
kubectl run test-pull --rm -i --restart=Never \
  --image=ghcr.io/your-org/edwin-api:latest \
  -n edwin \
  -- echo "Image pulled successfully"
```

### Pending Pods
```bash
# Check why pending
kubectl describe pod edwin-api-xxx -n edwin | grep -A 5 Events

# Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check PVC status
kubectl get pvc -n edwin
```

### Networking Issues
```bash
# Test service connectivity
kubectl run curl-test --rm -i --restart=Never \
  --image=nicolaka/netshoot:latest \
  -n edwin \
  -- curl -v http://edwin-api:8000/health

# Check DNS resolution
kubectl run dns-test --rm -i --restart=Never \
  --image=busybox:latest \
  -n edwin \
  -- nslookup edwin-api.edwin.svc.cluster.local

# Check endpoints
kubectl get endpoints -n edwin
```

---

## 📦 Batch Operations

### Apply Multiple Files
```bash
# Apply directory
kubectl apply -f k8s/

# Apply with kustomize
kubectl apply -k k8s/overlays/production

# Delete directory
kubectl delete -f k8s/
```

### Bulk Actions
```bash
# Scale all deployments
for deployment in $(kubectl get deployments -n edwin -o name); do
  kubectl scale $deployment --replicas=0 -n edwin
done

# Restart all deployments
for deployment in $(kubectl get deployments -n edwin -o name); do
  kubectl rollout restart $deployment -n edwin
done

# Delete all pods with specific label
kubectl delete pods -n edwin -l tier=backend
```

---

## 🎯 Production Checklist

### Pre-Deployment
```bash
# 1. Check cluster health
kubectl get nodes
kubectl get pods --all-namespaces

# 2. Verify secrets exist
kubectl get secrets -n edwin

# 3. Dry-run deployment
kubectl apply -f k8s/ --dry-run=client

# 4. Check resource quotas
kubectl describe resourcequota -n edwin
```

### Post-Deployment
```bash
# 1. Verify pods are running
kubectl get pods -n edwin

# 2. Check rollout status
kubectl rollout status deployment/edwin-api -n edwin

# 3. Test health endpoints
kubectl run health-test --rm -i --restart=Never \
  --image=curlimages/curl:latest \
  -n edwin \
  -- curl -s http://edwin-api:8000/health

# 4. Check logs for errors
kubectl logs -l app=edwin-api -n edwin --tail=100 | grep -i error

# 5. Monitor metrics
kubectl top pods -n edwin
```

---

## 🆘 Emergency Commands

### Quick Rollback
```bash
./scripts/rollback.sh production
```

### Scale to Zero (Emergency Stop)
```bash
kubectl scale deployment --all --replicas=0 -n edwin
```

### Get All Logs
```bash
kubectl logs -n edwin --all-containers=true \
  --since=1h > edwin-logs-$(date +%Y%m%d-%H%M%S).txt
```

### Emergency Debug Pod
```bash
kubectl run debug --rm -i --tty \
  --image=nicolaka/netshoot:latest \
  -n edwin \
  -- /bin/bash
```

---

## 📚 Useful Aliases

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias k='kubectl'
alias kgp='kubectl get pods -n edwin'
alias kgs='kubectl get svc -n edwin'
alias kgd='kubectl get deployments -n edwin'
alias kl='kubectl logs -f -n edwin'
alias kex='kubectl exec -it -n edwin'
alias kdesc='kubectl describe -n edwin'
alias kapp='kubectl apply -f'
alias kdel='kubectl delete -f'
```

---

**Quick Reference**: Keep this handy for day-to-day operations!
