"""
BDR Workflow Deployment Validation Script

Quick validation that all required components are available
for the BDR workflow deployment.
"""

import sys
import os

def check_imports():
    """Validate all required imports"""
    print("="*70)
    print("BDR WORKFLOW DEPLOYMENT VALIDATION")
    print("="*70)
    print()

    checks = []

    # Check 1: Core HoloLoom
    try:
        from HoloLoom.config import Config
        checks.append(("Config", True, "OK"))
    except Exception as e:
        checks.append(("Config", False, str(e)))

    # Check 2: Agentic system
    try:
        from HoloLoom.agentic import AgenticOrchestrator, ReasoningMode
        checks.append(("AgenticOrchestrator", True, "OK"))
    except Exception as e:
        checks.append(("AgenticOrchestrator", False, str(e)))

    # Check 3: Policy/Thompson Sampling
    try:
        from HoloLoom.policy.unified import create_policy, BanditStrategy
        checks.append(("Thompson Sampling", True, "OK"))
    except Exception as e:
        checks.append(("Thompson Sampling", False, str(e)))

    # Check 4: Alignment framework
    try:
        from HoloLoom.alignment import SafetyGuardrails, AuditTrail
        checks.append(("Safety Guardrails", True, "OK"))
    except Exception as e:
        checks.append(("Safety Guardrails", False, str(e)))

    # Check 5: Recursive learning
    try:
        from HoloLoom.recursive import FullLearningEngine
        checks.append(("Recursive Learning", True, "OK"))
    except Exception as e:
        checks.append(("Recursive Learning", False, str(e)))

    # Check 6: Types
    try:
        from HoloLoom.documentation.types import Query, MemoryShard
        checks.append(("Types", True, "OK"))
    except Exception as e:
        checks.append(("Types", False, str(e)))

    # Print results
    print("COMPONENT CHECKS:")
    print("-" * 70)
    for name, success, msg in checks:
        status = "[OK]" if success else "[FAIL]"
        print(f"{status:8} {name:30} {msg}")
    print()

    return all(success for _, success, _ in checks)


def check_files():
    """Validate required files exist"""
    print("FILE CHECKS:")
    print("-" * 70)

    files = [
        ("BDR Workflow JSON", "HoloLoom/web_dashboard/example_workflows/bdr_outbound_sequence.json"),
        ("Workflow Executor", "HoloLoom/web_dashboard/workflow_executor.py"),
        ("Demo Script", "demos/demo_bdr_workflow.py"),
        ("Implementation Guide", "BDR_WORKFLOW_GUIDE.md"),
        ("Quick Reference", "BDR_QUICK_REFERENCE.md"),
        ("Checklist", "BDR_IMPLEMENTATION_CHECKLIST.md"),
    ]

    all_exist = True
    for name, path in files:
        exists = os.path.exists(path)
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status:12} {name:30} {path}")
        all_exist = all_exist and exists

    print()
    return all_exist


def check_dependencies():
    """Check optional dependencies"""
    print("OPTIONAL DEPENDENCIES:")
    print("-" * 70)

    deps = []

    # FastAPI (required for workflow executor)
    try:
        import fastapi
        import uvicorn
        deps.append(("FastAPI/Uvicorn", True, "Required for workflow executor"))
    except:
        deps.append(("FastAPI/Uvicorn", False, "pip install fastapi uvicorn"))

    # Prometheus (optional)
    try:
        import prometheus_client
        deps.append(("Prometheus", True, "Metrics available"))
    except:
        deps.append(("Prometheus", False, "Optional: pip install prometheus_client"))

    # SendGrid (for email sending)
    try:
        import sendgrid
        deps.append(("SendGrid", True, "Email sending available"))
    except:
        deps.append(("SendGrid", False, "Optional: pip install sendgrid"))

    for name, available, msg in deps:
        status = "[OK]" if available else "[OPTIONAL]"
        print(f"{status:12} {name:30} {msg}")

    print()
    return True  # Optional deps don't fail validation


def main():
    # Run all checks
    imports_ok = check_imports()
    files_ok = check_files()
    deps_ok = check_dependencies()

    # Summary
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    if imports_ok and files_ok:
        print("[SUCCESS] All required components available!")
        print()
        print("NEXT STEPS:")
        print("1. Start workflow executor:")
        print("   cd HoloLoom/web_dashboard && python workflow_executor.py")
        print()
        print("2. Open workflow builder:")
        print("   http://localhost:8001/workflow_builder.html")
        print()
        print("3. Import BDR workflow:")
        print("   Click 'Import' -> Select example_workflows/bdr_outbound_sequence.json")
        print()
        return 0
    else:
        print("[FAILURE] Some components missing or unavailable")
        print()
        if not imports_ok:
            print("Fix import errors above before proceeding")
        if not files_ok:
            print("Some required files are missing")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())