# HoloLoom Environment Setup Guide

## Overview

HoloLoom implements environment-aware safety controls that automatically adjust based on your deployment environment. This allows for fast iteration in development while maintaining strict safety in production.

## Three Deployment Environments

### 1. DEVELOPMENT (Default)

**Purpose**: Local development and testing

**Safety Behavior**:
- **Auto-approve all actions** - No human approval required
- **Verbose logging** - DEBUG level (show everything)
- **All action categories auto-approved**: query, retrieval, analysis, storage, modification, execution, external

**Use When**:
- Local development
- Testing and debugging
- Rapid iteration
- Learning the system

**Risks**: No safety checks - suitable only for trusted local environments

### 2. STAGING

**Purpose**: Pre-production testing and validation

**Safety Behavior**:
- **Selective auto-approval** - Only safe read-only actions
- **Moderate logging** - INFO level
- **Auto-approved categories**: query, retrieval, analysis
- **Requires approval**: storage, modification, execution, external, deletion, system

**Use When**:
- Pre-production testing
- Integration testing
- Staging deployments
- Quality assurance

**Benefits**: Balances speed with safety - read operations are fast, write operations are gated

### 3. PRODUCTION

**Purpose**: Live deployment serving real users

**Safety Behavior**:
- **Require approval for all high-risk actions**
- **Minimal logging** - WARNING level (errors and warnings only)
- **No auto-approved categories** - All high-risk actions require human approval
- **Strict safety**: deletion and system actions always blocked

**Use When**:
- Production deployments
- Serving real users
- High-stakes environments
- Compliance-critical systems

**Benefits**: Maximum safety - all risky actions require explicit human approval

## Setting Your Environment

### Method 1: Environment Variable (Recommended)

Set the `HOLOLOOM_ENV` environment variable before starting the server:

**Windows (PowerShell)**:
```powershell
$env:HOLOLOOM_ENV = "development"
python start_agentic_server.py
```

**Windows (CMD)**:
```cmd
set HOLOLOOM_ENV=development
python start_agentic_server.py
```

**Linux/Mac (Bash)**:
```bash
export HOLOLOOM_ENV=development
python start_agentic_server.py
```

**Accepted Values**:
- `development`, `dev` → DEVELOPMENT
- `staging`, `stage` → STAGING
- `production`, `prod` → PRODUCTION

### Method 2: Programmatic Configuration

Set the environment directly in your code:

```python
from HoloLoom.config import Config, Environment

# Create config
config = Config.fast()

# Set environment explicitly
config.environment = Environment.DEVELOPMENT  # or STAGING, PRODUCTION

# Server will use this environment
```

### Method 3: Docker/Kubernetes

In `docker-compose.yml`:
```yaml
services:
  hololoom:
    environment:
      - HOLOLOOM_ENV=production
    # ... rest of config
```

In Kubernetes deployment:
```yaml
env:
  - name: HOLOLOOM_ENV
    value: "production"
```

## Verification

When the server starts, you'll see environment detection in the logs:

**Example (DEVELOPMENT)**:
```
INFO:HoloLoom.server.agentic_api_integrated:  Environment: development (from HOLOLOOM_ENV)
INFO:HoloLoom.server.agentic_api_integrated:  Safety mode: auto-approve all
```

**Example (PRODUCTION)**:
```
INFO:HoloLoom.server.agentic_api_integrated:  Environment: production (from HOLOLOOM_ENV)
INFO:HoloLoom.server.agentic_api_integrated:  Safety mode: require approvals
```

## Safety Behavior by Environment

| Action Category | DEVELOPMENT | STAGING | PRODUCTION |
|----------------|-------------|---------|------------|
| **QUERY** | ✓ Auto-approve | ✓ Auto-approve | ⚠ Evaluate |
| **RETRIEVAL** | ✓ Auto-approve | ✓ Auto-approve | ⚠ Evaluate |
| **ANALYSIS** | ✓ Auto-approve | ✓ Auto-approve | ⚠ Evaluate |
| **STORAGE** | ✓ Auto-approve | 🔒 Require approval | 🔒 Require approval |
| **MODIFICATION** | ✓ Auto-approve | 🔒 Require approval | 🔒 Require approval |
| **EXECUTION** | ✓ Auto-approve | 🔒 Require approval | 🔒 Require approval |
| **EXTERNAL** | ✓ Auto-approve | 🔒 Require approval | 🔒 Require approval |
| **DELETION** | ✓ Auto-approve | 🛑 Blocked | 🛑 Blocked |
| **SYSTEM** | ✓ Auto-approve | 🛑 Blocked | 🛑 Blocked |

**Legend**:
- ✓ Auto-approve: Action proceeds without human intervention
- ⚠ Evaluate: Action is evaluated against risk level, may require approval
- 🔒 Require approval: Action requires human approval before proceeding
- 🛑 Blocked: Action is blocked by default (critical risk)

## Logging Levels by Environment

| Environment | Level | What's Logged |
|------------|-------|---------------|
| DEVELOPMENT | DEBUG | Everything - full visibility into system operation |
| STAGING | INFO | Moderate detail - key operations and warnings |
| PRODUCTION | WARNING | Errors and warnings only - minimal noise |

**Access Logging Level**:
```python
from HoloLoom.config import Config

config = Config.fast()
config.environment = Environment.PRODUCTION
logging_level = config.logging_level  # Returns "WARNING"
```

## Advanced: Custom Safety Rules

You can customize safety behavior per environment by creating custom guardrails:

```python
from HoloLoom.alignment.safety_guardrails import SafetyGuardrails, SafetyPolicy
from HoloLoom.config import Config, Environment

# Create config
config = Config.fast()
config.environment = Environment.STAGING

# Create custom policy with additional auto-approve categories
custom_policy = SafetyPolicy(
    testing_mode=False,
    auto_approve_categories={"query", "retrieval", "analysis", "storage"}  # Also auto-approve storage
)

# Create guardrails with custom policy
guardrails = SafetyGuardrails(policy=custom_policy)

# Use with orchestrator
```

## Troubleshooting

### Issue: All actions being blocked in DEVELOPMENT

**Symptom**: Actions requiring approval even in development mode

**Solution**:
1. Verify `HOLOLOOM_ENV=development` is set
2. Check server logs for environment detection message
3. Restart server after setting environment variable

### Issue: Unknown environment warning

**Symptom**: `Unknown HOLOLOOM_ENV value: xxx, defaulting to DEVELOPMENT`

**Solution**: Use one of the accepted values:
- `development` or `dev`
- `staging` or `stage`
- `production` or `prod`

### Issue: Environment not detected

**Symptom**: No environment detection message in logs

**Solution**:
1. Ensure you're using the updated server code
2. Check that `_detect_environment()` function exists in server file
3. Verify import: `from HoloLoom.config import Environment`

## Best Practices

1. **Development**: Use DEVELOPMENT for all local work
   ```bash
   export HOLOLOOM_ENV=development
   ```

2. **CI/CD Testing**: Use STAGING for integration tests
   ```bash
   export HOLOLOOM_ENV=staging
   ```

3. **Production**: Always use PRODUCTION for live deployments
   ```bash
   export HOLOLOOM_ENV=production
   ```

4. **Docker**: Set environment in compose file
   ```yaml
   environment:
     - HOLOLOOM_ENV=${DEPLOY_ENV:-production}
   ```

5. **Verification**: Always check server logs for environment confirmation

## Security Notes

1. **Never use DEVELOPMENT in production** - No safety checks are performed
2. **STAGING is not production-safe** - Read operations bypass approval
3. **PRODUCTION is the only production-ready mode** - All high-risk actions gated
4. **Audit logs**: All safety decisions are logged regardless of environment
5. **Approval flow**: Production should integrate human-in-the-loop approval system

## Migration Guide

### From No Environment Awareness

If you're upgrading from a version without environment awareness:

**Before**:
```python
config = Config.fast()
# Safety behavior was global
```

**After**:
```python
config = Config.fast()
config.environment = Environment.DEVELOPMENT  # or set HOLOLOOM_ENV
# Safety behavior now environment-specific
```

**No Breaking Changes**: Default is DEVELOPMENT, existing behavior preserved for local development.

## Related Documentation

- [SAFETY_GUARDRAILS_OPTIONS_DETAILED.md](SAFETY_GUARDRAILS_OPTIONS_DETAILED.md) - Full safety system documentation
- [CLAUDE.md](CLAUDE.md) - General HoloLoom documentation
- [HoloLoom/alignment/safety_guardrails.py](HoloLoom/alignment/safety_guardrails.py) - Implementation reference
- [HoloLoom/config.py](HoloLoom/config.py) - Configuration reference

## Support

If you encounter issues:

1. Check server logs for environment detection message
2. Verify `HOLOLOOM_ENV` is set correctly
3. Ensure imports are up-to-date
4. Review audit logs at `./alignment_logs/` for safety decisions

## Summary

**Quick Reference**:

```bash
# Development (fast iteration, no approvals)
export HOLOLOOM_ENV=development

# Staging (read fast, write gated)
export HOLOLOOM_ENV=staging

# Production (all high-risk actions gated)
export HOLOLOOM_ENV=production
```

The environment-aware safety system ensures you can move fast in development while maintaining strict safety in production. Choose the right environment for your use case and verify the server logs confirm your selection.
