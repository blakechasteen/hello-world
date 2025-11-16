# DDoS Attack Incident Runbook

**Version**: 1.0.0
**Created**: 2025-11-16
**Severity**: SEV-2 (HIGH) or SEV-1 (if complete outage)
**Type**: Availability / Denial of Service

---

## Quick Start (First 10 Minutes)

```
1. ALERT: "DDoS Attack Detected" or unusual traffic spike
2. VERIFY: Is traffic legitimate spike or actual attack?
3. CLASSIFY: What % of traffic is from attack? (determines SEV)
4. PAGE: Incident Commander, Network Security Lead
5. ACTIVATE: DDoS mitigation (Cloudflare, AWS Shield, etc)
6. BLOCK: Attacker traffic at firewall/WAF
7. SCALE: Increase capacity if needed (auto-scaling)
8. MONITOR: Track attack progression
```

---

## Detection & Analysis

### Identify Attack Type

**Volumetric Attack** (target bandwidth):
- High volume of traffic from many sources (botnet)
- SIGKILL attack: 100+ Gbps of junk traffic
- Flood attack: UDP, ICMP, or DNS floods
- Amplification attack: Attacker spoofs source IP
- Goal: Overwhelm network pipes

**Protocol Attack** (consume resources):
- SYN flood: Attacker sends many TCP SYN packets (half-open connections)
- Fragmented packet attacks
- Ping of death
- Goal: Overwhelm OS network stack

**Application Attack** (exhaust web server):
- HTTP request flood (many requests to single URL)
- Slowloris: Send requests slowly (keep connections open)
- Goal Layer: App (HTTP/HTTPS), harder to distinguish from legitimate traffic

### Indicators

```
Volumetric Indicators:
[ ] Network traffic spike (10x+ normal)
[ ] Bandwidth utilization near 100%
[ ] Unique source IPs: Thousands+
[ ] Geographically diverse sources
[ ] Traffic from known botnet ranges

Protocol Attack Indicators:
[ ] SYN queue depth high
[ ] Many half-open connections (netstat | grep SYN_RECV | wc -l)
[ ] CPU high on network interfaces
[ ] OS-level metrics degraded

Application Attack Indicators:
[ ] HTTP request volume spike
[ ] Requests to specific URL (less distributed)
[ ] User-agent strings identical across requests
[ ] Geographic distribution more concentrated
[ ] Can be harder to distinguish from legitimate spike
```

---

## Immediate Response (T+0 to T+15 min)

### Step 1: Activate DDoS Mitigation

```
[ ] If using Cloudflare:
    - Set Zone > Security > DDoS Protection Level to "I'm Under Attack"
    - Enable Rate Limiting rule (threshold: based on normal traffic)
    - Monitor analytics in real-time
    - Consider enabling CAPTCHA challenge

[ ] If using AWS Shield Advanced:
    - Check AWS DRT (DDoS Response Team) ready
    - Verify auto-scaling policies active
    - Check CloudFront and WAF rules

[ ] If using other provider:
    - Activate DDoS mitigation mode
    - Ensure traffic being scrubbed
    - Verify legitimate traffic can reach you
```

### Step 2: Implement Rate Limiting & Blocking

```
Firewall Rules (block obvious attack sources):
[ ] Block IP ranges known for botnets
[ ] Block traffic from attacker source (if visible)
[ ] Rate limit per IP: 100 requests per second (adjust as needed)
[ ] Block if User-Agent is blank or bot-like
[ ] Block if Referer is suspicious

WAF Rules:
[ ] Enable geo-blocking (if attack from specific country)
[ ] Enable bot detection / reCAPTCHA
[ ] Rate limit by IP: 1000 requests/min
[ ] Block if no valid browser headers
[ ] Enable request counting / CAPTCHA on spike
```

### Step 3: Scale Infrastructure

```
[ ] Check auto-scaling group status
    - Min capacity: Increase if needed
    - Max capacity: Ensure not capped
    - Target utilization: Adjust if needed

[ ] Add additional capacity
    - Spin up extra web servers
    - Scale database read replicas
    - Increase CDN edge location capacity

[ ] Verify cache strategy
    - Cache static content (CSS, JS, images)
    - Cache API responses if appropriate
    - Offload CPU-intensive operations
```

---

## Investigation (T+15 min to T+2 hours)

### Determine Attack Characteristics

```
Attack Source:
[ ] Is traffic from single IP or distributed?
[ ] Geographic origin of attack
[ ] ASN/Netblock of attacking IPs
[ ] Is source IP spoofed or real?

Attack Target:
[ ] Targeted at specific URL/endpoint?
[ ] Or general bandwidth saturation?
[ ] Specific port(s) or all ports?
[ ] Specific protocols (HTTP, TCP, UDP)?

Attack Volume:
[ ] Peak traffic rate (Gbps or requests/sec)
[ ] Baseline normal traffic
[ ] Attack percentage of total traffic
[ ] Attack duration (minutes or hours)

Attack Pattern:
[ ] Is attack constant or waves?
[ ] Does it change tactics (volumetric → protocol → application)?
[ ] Correlation with news events or social media?
```

### Determine if Extortion/Targeted

```
[ ] Has attacker made extortion demands?
    - Email, Slack message, social media
    - Bitcoin wallet for "protection"
    - Usually ignored (don't engage)

[ ] Is attack tied to competitor/rival?
    - Unusual timing (competitive event)?
    - Targeted at specific functionality?
    - Previous threats/warnings?

[ ] Is it random opportunistic attack?
    - No clear motive
    - Targeting vulnerable/visible services
    - Low sophistication

Important: Do NOT negotiate or pay ransom (funds crime, no guarantee attack stops)
```

---

## Containment (T+30 min to T+4 hours)

### Block Attacker Traffic

```
By IP Address:
[ ] Identify attacking IP ranges
[ ] Add firewall rules to drop traffic
[ ] Verify legitimate traffic not blocked
[ ] Monitor for new attacking IPs (adjust rules)

By Geographic Region:
[ ] If attack from specific country:
    - Implement geo-blocking (if service not needed there)
    - Example: Block traffic from country X if you're US-only
    - Verify legitimate users in that country are few

By Behavioral Patterns:
[ ] Block traffic with no User-Agent header
[ ] Block with invalid HTTP headers
[ ] Block traffic pattern matching known bots
[ ] Require JavaScript execution (blocks basic bots)
```

### Degrade Gracefully

```
If Attack Continues & Overwhelming:

Option 1: Request Whitelisting
[ ] Add legitimate IPs/ranges to whitelist
[ ] Block all other traffic (drastic but effective)
[ ] Users provide IP, get added to whitelist
[ ] Risk: Legitimate users unable to access

Option 2: Rate Limiting + CAPTCHA
[ ] Reduce rate limit threshold (fewer requests allowed)
[ ] Require CAPTCHA after threshold exceeded
[ ] Discourages bots, slows humans slightly
[ ] Reduces impact but not total protection

Option 3: Service Degradation
[ ] Reduce feature complexity temporarily
[ ] Cache everything possible
[ ] Disable non-critical features
[ ] Offline batch processing instead of real-time
[ ] Provide status page updates

Option 4: Redirect to Backup
[ ] If primary datacenter overwhelmed
[ ] Failover to backup datacenter
[ ] May have slight performance impact
[ ] Better than complete outage
```

---

## Recovery (T+4 hours to T+24 hours)

### Monitor for Attack Cessation

```
Traffic Monitoring:
[ ] Track attack traffic volume over time
[ ] Is it declining? (good sign)
[ ] Stabilizing? (attacker getting bored)
[ ] Switching tactics? (prepare for new attack type)

Real-Time Dashboards:
[ ] Traffic distribution (legitimate vs. attack)
[ ] Geographic origin of traffic
[ ] Top attacking IPs
[ ] Request rate trends

Alert Configuration:
[ ] Set alert if attack resumes
[ ] Alert if attack volume increases
[ ] Monitor for re-compromised infrastructure (if botnet part of compromise)
```

### Return to Normal Operations

```
Gradual Recovery:
[ ] Monitor normal traffic patterns
[ ] Gradually reduce rate limiting thresholds
[ ] Remove temporary blocking rules
[ ] Verify all services operational

Performance Check:
[ ] Test critical functionality
[ ] Verify database consistency
[ ] Check for stuck background jobs
[ ] Review error logs for artifacts

Maintenance:
[ ] Clear attack logs (preserve for forensics)
[ ] Update firewall/WAF rules with lessons learned
[ ] Review auto-scaling policies (were they effective?)
[ ] Plan infrastructure improvements
```

---

## Post-Mortem (Day 3)

### Lessons Learned

```
Detection:
[ ] How quickly was attack detected? (5 min? 30 min? Hours?)
[ ] Could detection be faster?
[ ] Were alerts actionable?

Response:
[ ] Were mitigation services effective?
[ ] Was traffic actually blocked?
[ ] Did rate limiting work?
[ ] Did auto-scaling respond in time?

Mitigation Effectiveness:
[ ] Did we maintain <X% impact on users?
[ ] How long until 99% of traffic normal?
[ ] Were legitimate users impacted? How many?
[ ] Customer complaints?
```

### Improvements

```
Infrastructure:
[ ] Add DDoS mitigation service if not already
[ ] Increase network capacity buffers
[ ] Implement geo-redundant infrastructure
[ ] Test failover to backup datacenters

Detection:
[ ] Improve baseline traffic profiling
[ ] Add more granular traffic monitoring
[ ] Implement anomaly detection
[ ] Set lower alert thresholds

Process:
[ ] Document attack timeline better
[ ] Improve on-call notification process
[ ] Reduce time to activate mitigation
[ ] Better customer communication
```

---

## Quick Reference

```
SEVERITY:
SEV-2: If <50% of traffic affected or <15 min downtime
SEV-1: If >50% traffic or >15 min complete outage

ATTACK TYPES:
Volumetric: Bandwidth saturation (Cloudflare/AWS Shield effective)
Protocol: Network stack attack (firewall rules effective)
Application: HTTP floods (rate limiting + CAPTCHA effective)

IMMEDIATE ACTIONS (5 min):
1. Activate DDoS mitigation service
2. Block obvious attack traffic
3. Scale infrastructure
4. Page incident commander

ESCALATION:
SEV-2: Page network team
SEV-1: Wake incident commander, CTO, possible press statement

INVESTIGATION (30 min):
- Attack source (single IP? botnet?)
- Attack target (specific URL? bandwidth?)
- Attack volume (Gbps? requests/sec?)
- Is it extortion? (don't pay)

CONTAINMENT (1-4 hours):
[ ] Block attacking IPs
[ ] Rate limit per IP
[ ] Geo-block if applicable
[ ] Degrade gracefully if needed

RECOVERY (4-24 hours):
[ ] Monitor attack volume
[ ] Gradually reduce rate limiting
[ ] Return to normal operations
[ ] Check system integrity

POST-MORTEM (Day 3):
[ ] How fast detected?
[ ] How effective was mitigation?
[ ] What to improve for next time?
[ ] Update procedures
```

---

**Status**: ✅ Production Ready (2025-11-16)
**Last Tested**: 2025-10-15 (DDoS simulation drill)
**Next Drill**: 2026-01-15
**Owner**: Network Security Lead
