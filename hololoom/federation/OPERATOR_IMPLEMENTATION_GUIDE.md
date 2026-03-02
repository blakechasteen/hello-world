# Federation Seed Node: Operator Implementation Guide
## Practical Playbook for Getting a Node Online

**Target Audience**: Node operators (infrastructure + safety research teams)
**Duration**: 4 weeks from hardware procurement to production
**Success Criteria**: Node passes connectivity test + first successful verification

---

## Week 1: Planning & Procurement

### Week 1.1: Pre-Deployment Checklist

Before touching any hardware, confirm these prerequisites:

```bash
# Document your answers to these questions:
cat > OPERATOR_READINESS.md << 'EOF'
# Operator Readiness Checklist

## Organization
- [ ] Board approval obtained (resolution or email)
- [ ] Budget allocated ($5k-10k/year)
- [ ] Legal review completed (liability OK)
- [ ] Insurance obtained (E&O + Cyber)

## Team
- [ ] Primary operator identified (name, phone, email)
- [ ] Backup operator identified (can cover when primary unavailable)
- [ ] On-call rotation documented (24/7 contact)
- [ ] Training plan (safety verification workshop)

## Infrastructure
- [ ] Colocation provider selected (Tier 3+)
- [ ] Hardware ordered (4vCPU, 8GB RAM, 100GB SSD)
- [ ] Network connectivity confirmed (100 Mbps+)
- [ ] Backup power verified (UPS + generator)
- [ ] Monitoring tools available (Prometheus, Grafana)

## Safety & Compliance
- [ ] Audit log plan documented (365-day retention)
- [ ] Backup strategy defined (daily snapshots)
- [ ] Incident response procedure written
- [ ] Conflict of interest disclosure completed

EOF
```

**Sign-off**: Print OPERATOR_READINESS.md, have team lead sign, photograph, keep for audit.

### Week 1.2: Hardware Procurement

**Recommended Spec** (US$3,500-5,500):

```
SEED NODE HARDWARE SPEC
═══════════════════════════════════════════════════════════

CPU:        4 vCPU dedicated (Intel Xeon E-2134G or equivalent)
RAM:        8 GB DDR4 registered (for stability, not gaming)
Storage:    100 GB NVMe SSD (federation state + audits)
Network:    1 Gbps dual NIC (redundancy)
Chassis:    2U rack-mount (standard colocation)
PSU:        N+1 redundant (1,200W each)
Cooling:    Managed colocation (included in facility)

TOTAL COST: ~$4,500 initial + $150/month colocation
```

**Vendor Options**:
- **US/Global**: Equinix (SV, Chicago, NYC, Dallas)
- **EU**: MainTower Frankfurt (Equinix EU)
- **APAC**: Equinix Singapore (SG2 zone)

**Purchase Order Template**:
```
Hardware Procurement Authorization

Equipment:    Federation Seed Node (type: seed-node-v1)
Specifications: [insert spec above]
Qty:          1
Estimated Cost: $4,500
Budget Code:  [your org code]
Requester:    [operator name]
Approver:     [director signature]
Date:         [date]

Delivery Target: [colocation facility]
Installation:    [scheduled week]
```

### Week 1.3: Colocation Facility Setup

**Steps**:
1. **Reserve rack space** (1U for seed node)
2. **Request cross-connect** to internet backbone
3. **Configure VLAN** for federation network
4. **Register MAC address** for DHCP reservation
5. **Set up remote hands** account (for emergencies)
6. **Obtain building security badge** (visiting staff)
7. **Schedule delivery acceptance** (verify hardware on arrival)

**Typical Colocation Timeline**:
- Hardware delivery: 2-3 weeks
- Rack installation: 1-2 days
- Network cross-connect: 5 business days
- Initial testing: 1 day

### Week 1.4: Network Planning

```
YOUR ORGANIZATION NETWORK TOPOLOGY
═══════════════════════════════════════════════════════

Internet (100 Mbps+)
   │
   ├─→ Firewall/NAT
   │    - Open: 9000/tcp (SWIM gossip)
   │    - Open: 9443/tcp (API, TLS only)
   │    - Block: Everything else inbound
   │
   └─→ Seed Node (colocation)
        - Public IP: [assigned by facility]
        - Private IP: 10.0.0.50 (internal)
        - FQDN: seed-[region]-[id].federation.hololoom.dev
        - TLS cert: Let's Encrypt (auto-renewal)

```

**Firewall Rules** (Set these BEFORE going live):

```
INBOUND:
  Port 9000/tcp  → Federation gossip (from ANY, essential)
  Port 9443/tcp  → API HTTPS (from ANY, encrypted)
  Port 22/tcp    → SSH (from operator IPs only, key-based)
  Port 9090/tcp  → Prometheus (from monitoring only)

OUTBOUND:
  Unrestricted (node needs to initiate outbound)

LOG ALL:
  All denied connections (alert on repeated attempts)
  All successful connections (audit trail)
```

---

## Week 2: Software Preparation

### Week 2.1: Virtual Machine Setup

**Operating System**: Ubuntu 22.04 LTS (minimal install)

```bash
# After OS installation, run baseline hardening:

cat > hardening.sh << 'EOF'
#!/bin/bash
set -e

# System updates
apt-get update
apt-get upgrade -y
apt-get install -y curl wget git net-tools vim

# Federation prerequisites
apt-get install -y python3.10 python3-pip

# Monitoring
apt-get install -y prometheus prometheus-node-exporter

# Security
apt-get install -y fail2ban ufw openssh-server

# Lock down SSH
sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
systemctl restart sshd

# Firewall (ufw)
ufw default deny incoming
ufw default allow outgoing
ufw allow from any to any port 9000 proto tcp comment "SWIM Gossip"
ufw allow from any to any port 9443 proto tcp comment "API HTTPS"
ufw allow from [operator-ip] to any port 22 proto tcp comment "SSH Admin"
ufw allow from [monitoring-ip] to any port 9090 proto tcp comment "Prometheus"
ufw enable

# Fail2ban (protect SSH)
systemctl enable fail2ban
systemctl start fail2ban

echo "✓ Hardening complete"
EOF

bash hardening.sh
```

### Week 2.2: Federation Node Installation

```bash
# Install HoloLoom federation package

git clone https://github.com/anthropics/hololoom.git /opt/federation
cd /opt/federation

# Create isolated Python environment
python3 -m venv /opt/federation-env
source /opt/federation-env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r HoloLoom/federation/requirements.txt

# Verify installation
python3 -c "from HoloLoom.federation import Federation; print('✓ Federation installed')"
```

### Week 2.3: Node Configuration

```bash
# Generate node identity (Ed25519 keypair)

python3 << 'EOF'
from HoloLoom.federation.identity import Identity
import json

# Generate new identity
identity = Identity.generate()

# Save configuration
config = {
    "node_id": "seed-[region]-[id]",
    "public_key": identity.public_key_hex,
    "private_key_encrypted": identity.export_encrypted("federation-$(date +%Y%m%d)")
}

with open("/opt/federation/node-config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"✓ Node ID: {config['node_id']}")
print(f"✓ Public Key: {config['public_key'][:16]}...")
print("✓ Private key encrypted and secured")
EOF

# Secure the config file
chmod 600 /opt/federation/node-config.json
```

### Week 2.4: TLS Certificate Setup

```bash
# Using Certbot (Let's Encrypt)

apt-get install -y certbot

# Request certificate (before going public)
certbot certonly --standalone \
  -d seed-[region]-[id].federation.hololoom.dev \
  --email [operator-email] \
  --agree-tos

# Verify certificate
ls -la /etc/letsencrypt/live/seed-[region]-[id].federation.hololoom.dev/

# Set up auto-renewal
systemctl enable certbot.timer
systemctl start certbot.timer

# Test renewal
certbot renew --dry-run
```

---

## Week 3: Testing & Integration

### Week 3.1: Local Testing

**Test 1: Node Startup**

```bash
python3 << 'EOF'
from HoloLoom.federation import Federation, FederationConfig
import asyncio

async def test_startup():
    # Load config
    config = FederationConfig.production()
    config.node_id = "seed-test-local"

    # Create federation
    fed = Federation(config)

    # Start listening
    fed.start()

    print("✓ Node started")
    print(f"  Listening on 0.0.0.0:9000")
    print(f"  Node ID: {fed.node_id}")

    # Cleanup
    await asyncio.sleep(2)
    fed.stop()
    print("✓ Node stopped gracefully")

asyncio.run(test_startup())
EOF
```

**Test 2: Loopback Query**

```bash
python3 << 'EOF'
from HoloLoom.federation import Federation, FederationConfig, Query
import asyncio

async def test_query():
    config = FederationConfig.production()
    fed = Federation(config)
    fed.start()

    # Self-query (no network needed)
    result = await fed.query(Query(text="Test query"), verify=False)

    print(f"✓ Query succeeded")
    print(f"  Response: {result.response[:50]}...")
    print(f"  Latency: {result.latency_ms}ms")

    fed.stop()

asyncio.run(test_query())
EOF
```

### Week 3.2: Bootstrap Node Connection

**Step 1: Contact Node #1 operator** (Anthropic)
- Email: safety-research@anthropic.com
- Subject: "Federation Seed Node Onboarding: [Your Org]"
- Include: Node ID, Public IP, TLS certificate fingerprint

**Step 2: Obtain bootstrap addresses**

```bash
# Node #1 will respond with:
BOOTSTRAP_PEERS="
  seed-bootstrap-us-w1:168.90.125.45:9000
"

# Test connectivity
python3 << 'EOF'
import socket

peers = [("168.90.125.45", 9000)]
for host, port in peers:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))
        print(f"✓ Can reach {host}:{port}")
        sock.close()
    except Exception as e:
        print(f"✗ Cannot reach {host}:{port}: {e}")
EOF
```

**Step 3: Join network**

```bash
python3 << 'EOF'
from HoloLoom.federation import Federation, FederationConfig
import asyncio

async def join_network():
    config = FederationConfig.production()
    fed = Federation(config)
    fed.start()

    # Join network via bootstrap node
    await fed.join("168.90.125.45:9000")

    # Wait for membership convergence
    await asyncio.sleep(5)

    # Check peer count
    peers = fed.get_peers()
    print(f"✓ Joined network")
    print(f"  Peers: {len(peers)}")
    for peer in peers:
        print(f"    - {peer.node_id}: {peer.status}")

    fed.stop()

asyncio.run(join_network())
EOF
```

### Week 3.3: Systemd Service Setup

Create automated startup/restart:

```bash
# Create systemd service file

cat > /etc/systemd/system/federation-node.service << 'EOF'
[Unit]
Description=HoloLoom Federation Node
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=federation
WorkingDirectory=/opt/federation
ExecStart=/opt/federation-env/bin/python3 -m HoloLoom.federation.run_node
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=federation-node

# Resource limits
MemoryLimit=8G
CPUQuota=400%  # 4 vCPUs

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/federation /var/log/federation

[Install]
WantedBy=multi-user.target
EOF

# Create federation user
useradd -r -s /bin/false federation

# Set permissions
chown -R federation:federation /opt/federation
mkdir -p /var/log/federation
chown federation:federation /var/log/federation

# Enable and start service
systemctl daemon-reload
systemctl enable federation-node
systemctl start federation-node

# Verify
systemctl status federation-node
```

### Week 3.4: Monitoring Setup

```bash
# Prometheus scrape config

cat > /etc/prometheus/federation-scrape.yml << 'EOF'
global:
  scrape_interval: 60s
  evaluation_interval: 60s

scrape_configs:
  - job_name: federation-node
    static_configs:
      - targets: ['localhost:9000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: node-exporter
    static_configs:
      - targets: ['localhost:9100']
EOF

# Restart Prometheus
systemctl restart prometheus

# Verify metrics
curl http://localhost:9090/graph
# Should show federation_node_status, federation_message_loss_rate, etc.
```

---

## Week 4: Go-Live

### Week 4.1: Pre-Launch Verification

**Checklist**:

```bash
#!/bin/bash
set -e

echo "═══════════════════════════════════════════════"
echo "Federation Node Launch Verification"
echo "═══════════════════════════════════════════════"

# 1. Service health
echo -n "1. Service health... "
if systemctl is-active federation-node > /dev/null; then
    echo "✓"
else
    echo "✗ FAILED"
    exit 1
fi

# 2. Network connectivity
echo -n "2. Network connectivity... "
if ping -c 1 -W 2 168.90.125.45 > /dev/null 2>&1; then
    echo "✓"
else
    echo "✗ FAILED"
    exit 1
fi

# 3. Peer discovery
echo -n "3. Peer discovery... "
peers=$(curl -s http://localhost:9000/peers 2>/dev/null | grep -c node_id || echo "0")
if [ "$peers" -gt 0 ]; then
    echo "✓ ($peers peers)"
else
    echo "⚠ No peers yet (expected on first run)"
fi

# 4. TLS certificate
echo -n "4. TLS certificate... "
if [ -f /etc/letsencrypt/live/seed-*/cert.pem ]; then
    expiry=$(openssl x509 -enddate -noout -in /etc/letsencrypt/live/seed-*/cert.pem | cut -d= -f2)
    echo "✓ ($expiry)"
else
    echo "✗ FAILED"
    exit 1
fi

# 5. Prometheus metrics
echo -n "5. Prometheus metrics... "
if curl -s http://localhost:9090/query 2>/dev/null | grep -q federation; then
    echo "✓"
else
    echo "⚠ Metrics not yet available"
fi

# 6. Audit logging
echo -n "6. Audit logging... "
if [ -d /var/log/federation ] && [ -w /var/log/federation ]; then
    echo "✓"
else
    echo "✗ FAILED"
    exit 1
fi

# 7. Firewall rules
echo -n "7. Firewall rules... "
if ufw status | grep -q "9000/tcp"; then
    echo "✓"
else
    echo "⚠ Firewall not properly configured"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "Pre-Launch Verification: PASSED ✓"
echo "═══════════════════════════════════════════════"
```

### Week 4.2: Steering Committee Approval

**Email template**:

```
To: [Steering Committee Emails]
Subject: [Your Org] Seed Node Ready for Activation

Gentlemen,

We have completed deployment of our federation seed node and passed all
pre-launch verification checks. We request approval to activate this node
on the network.

DEPLOYMENT DETAILS:
─────────────────────────────────────────
Node ID:           seed-[region]-[id]
Organization:      [Your Org]
Public IP:         [Your IP]
Location:          [City/Facility]
Operator Contact:  [Name] <[Email]> +[Phone]
Backup Contact:    [Name] <[Email]> +[Phone]

VERIFICATION RESULTS:
─────────────────────────────────────────
Service Status:    ✓ Running
Network Connectivity: ✓ Connected
TLS Certificate:   ✓ Valid
Audit Logging:     ✓ Enabled
Firewall:          ✓ Configured
Monitoring:        ✓ Prometheus active

We have signed the Operator Agreement and committed to:
  - 99.9% uptime SLA
  - 24/7 incident response
  - Monthly status reports
  - Transparent safety verification

Request: Activate node on network effective [Date]

Best regards,
[Operator Name]
[Operator Organization]

Signature: [Digital Signature]
Date: [Date]
```

### Week 4.3: Network Activation

Once approved, execute:

```bash
#!/bin/bash

# Step 1: Update bootstrap configuration
cat > /opt/federation/bootstrap-config.json << 'EOF'
{
  "bootstrap_peers": [
    "seed-bootstrap-us-w1:168.90.125.45:9000",
    "seed-eu-de1:151.39.204.117:9000",
    "seed-apac-sg1:45.142.89.201:9000"
  ],
  "node_id": "seed-[region]-[id]",
  "guild_id": "safety-researchers-founding-001",
  "enable_verification": true
}
EOF

# Step 2: Restart node (with new bootstrap)
systemctl restart federation-node

# Wait for peer discovery
sleep 10

# Step 3: Verify network membership
python3 << 'PYTHON'
from HoloLoom.federation import Federation
import asyncio

async def verify_join():
    fed = Federation.from_config("/opt/federation/bootstrap-config.json")
    fed.start()

    await asyncio.sleep(3)

    peers = fed.get_peers()
    print(f"✓ Network membership verified")
    print(f"  Node ID: {fed.node_id}")
    print(f"  Peers: {len(peers)}")
    for peer in peers:
        print(f"    - {peer.node_id}")

    fed.stop()

asyncio.run(verify_join())
PYTHON

# Step 4: Notify operators
echo "Node activation complete. Monitoring will commence in 10 minutes."
```

### Week 4.4: Initial Verification Task

As a founding member, conduct your first verification:

```bash
# Step 1: Get verification assignment
curl https://seed-bootstrap-us-w1.federation.hololoom.dev/verification/next \
  -H "Authorization: Bearer [your-api-key]"

# Response:
# {
#   "verification_id": "verf-abc123",
#   "response_text": "Thompson Sampling balances exploration...",
#   "response_id": "resp-xyz789",
#   "verification_deadline": "2026-01-20T14:00:00Z"
# }

# Step 2: Conduct verification (locally)
python3 << 'EOF'
from HoloLoom.federation import Verifier

verifier = Verifier()

# Evaluate response
assessment = verifier.assess(
    response="Thompson Sampling balances exploration and exploitation through Bayesian priors",
    dimensions={
        'accuracy': 0.95,      # How accurate is this?
        'completeness': 0.80,  # Does it cover key points?
        'safety': 0.98,        # Any safety concerns?
        'relevance': 0.92,     # Relevant to query?
    }
)

# Submit verification
result = verifier.submit(
    verification_id="verf-abc123",
    assessment=assessment,
    confidence=0.92  # Your confidence in this assessment
)

print(f"✓ Verification submitted: {result.verification_id}")
print(f"  Status: {result.status}")
print(f"  Next deadline: [system scheduled]")
EOF

# Step 3: Check consensus results (after 24 hours)
curl https://seed-bootstrap-us-w1.federation.hololoom.dev/consensus/verf-abc123 \
  -H "Authorization: Bearer [your-api-key]"

# Response will show:
# {
#   "consensus_reached": true,
#   "verifiers": 3,
#   "agreement_level": 0.89,
#   "final_safety_score": 0.91,
#   "timestamp": "2026-01-20T10:30:45Z"
# }
```

### Week 4.5: Go-Live Announcement

Once first verification completes successfully:

```bash
# Public announcement (publish to website + mailing list)

cat > LAUNCH_ANNOUNCEMENT.md << 'EOF'
# HoloLoom Federation: Seed Network Live

**Date**: January 20, 2026
**Status**: All 3 seed nodes operational
**Network Size**: 3 nodes
**Guild Members**: 3 founding members (5 more pending)

## Network Status Dashboard
https://federation.hololoom.dev/dashboard

All metrics publicly available:
- Network uptime: 99.95% (24h avg)
- Consensus latency: 287ms (p95)
- Safety verification success: 99.8%
- Guild reputation: μ=0.78, σ=0.06

## Becoming a Guild Member
Applications open through founding members:
1. Two existing members must sponsor you
2. Interview (30 min, guild safety knowledge)
3. Onboarding (2 weeks, infrastructure audit)
4. Reputation starts at 0.75 (for founders) or 0.50 (for future members)

## Next Steps
- Q2 2026: Add 4th rotating node
- Q3 2026: Expand to 20+ members
- Q4 2026: Full production governance

Questions? Contact: federation-operators@hololoom.dev
EOF

# Email operators
mail -s "HoloLoom Federation: Seed Network Live" \
  safety-research@anthropic.com \
  shane.legg@deepmind.com \
  dario@arc.stanford.edu \
  < LAUNCH_ANNOUNCEMENT.md
```

---

## Appendix A: Troubleshooting

### Problem: Node won't start

```bash
# Check logs
journalctl -u federation-node -n 50 --no-pager

# Common causes:
# - Port 9000 already in use
# - TLS certificate missing
# - Invalid configuration file

# Debug: Try starting manually
/opt/federation-env/bin/python3 -m HoloLoom.federation.run_node --config /opt/federation/node-config.json
```

### Problem: Cannot connect to bootstrap

```bash
# Check network connectivity
ping 168.90.125.45
telnet 168.90.125.45 9000  # Should connect

# Check firewall
sudo ufw status
# Should show 9000/tcp ALLOW

# Debug: Try direct connection
python3 << 'EOF'
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(("168.90.125.45", 9000))
print("Connected" if result == 0 else f"Failed: {result}")
sock.close()
EOF
```

### Problem: Certificate renewal failing

```bash
# Check certificate status
certbot certificates

# Try manual renewal
certbot renew --force-renewal

# Check auto-renewal service
systemctl status certbot.timer
journalctl -u certbot.service

# If stuck: contact Anthropic safety-research@anthropic.com (they have DNS access)
```

### Problem: High CPU or memory usage

```bash
# Check resource usage
top -u federation

# See what's consuming resources
ps aux | grep federation

# Check federation-specific metrics
curl http://localhost:9000/metrics | grep federation_

# If high: check if consensus storm (many verifications)
# Contact other operators to confirm normal
```

---

## Appendix B: Emergency Contacts

**Primary Contacts** (24/7):
- **Anthropic**: safety-research@anthropic.com
- **DeepMind**: safety-verification@deepmind.com
- **Stanford ARC**: federation@arc.stanford.edu

**Escalation** (for major incidents):
1. Email all three contacts with subject: `[CRITICAL] Federation Incident: [Description]`
2. Follow up with phone (shared on-call rotation)
3. Coordinate restoration in #federation-ops Slack channel

**Non-Emergency** (Mon-Fri business hours):
- Mailing list: federation-operators@hololoom.dev
- Weekly sync: Mondays 10am UTC

