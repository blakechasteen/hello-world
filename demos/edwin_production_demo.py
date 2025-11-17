"""
EdWIN Production Deployment Demo
Demonstrates authentication, database connectivity, and production features

Implementation Date: November 17, 2025

Usage:
    PYTHONPATH=. python demos/edwin_production_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from EduVerse.edwin.auth import (
    create_user, authenticate_user, create_access_token,
    create_refresh_token, decode_token, UserRole
)
from EduVerse.edwin.database import DatabaseManager


async def demo_authentication():
    """Demonstrate authentication flow"""
    print("\n" + "=" * 60)
    print("1. Authentication Demo")
    print("=" * 60)

    # Create test user
    print("\n📝 Creating test user...")
    user = await create_user(
        username="demo_student",
        email="demo@edwin.edu",
        password="DemoPassword123!@#",
        full_name="Demo Student",
        role=UserRole.STUDENT,
        student_id="demo_123"
    )
    print(f"✅ Created user: {user.username} (role: {user.role.value})")

    # Authenticate
    print("\n🔐 Authenticating...")
    authenticated = await authenticate_user("demo_student", "DemoPassword123!@#")

    if authenticated:
        print(f"✅ Authentication successful!")
        print(f"   User ID: {authenticated.user_id}")
        print(f"   Email: {authenticated.email}")
        print(f"   Role: {authenticated.role.value}")
    else:
        print("❌ Authentication failed")
        return

    # Generate tokens
    print("\n🔑 Generating JWT tokens...")
    access_token = create_access_token(
        user_id=authenticated.user_id,
        username=authenticated.username,
        role=authenticated.role
    )
    refresh_token = create_refresh_token(authenticated.user_id)

    print(f"✅ Access token: {access_token[:50]}...")
    print(f"✅ Refresh token: {refresh_token[:50]}...")

    # Decode token
    print("\n🔍 Decoding access token...")
    token_data = decode_token(access_token)
    print(f"✅ Token payload:")
    print(f"   User ID: {token_data.user_id}")
    print(f"   Username: {token_data.username}")
    print(f"   Role: {token_data.role.value}")
    print(f"   Expires: {token_data.exp}")


async def demo_database():
    """Demonstrate database connectivity"""
    print("\n" + "=" * 60)
    print("2. Database Connectivity Demo")
    print("=" * 60)

    # Initialize database manager
    print("\n🔧 Initializing database manager...")
    db = DatabaseManager(use_persistent_storage=False, enable_redis=False)
    await db.initialize()

    print("\n✅ Database manager initialized")
    print(f"   Knowledge graph: {type(db.knowledge_graph).__name__}")
    print(f"   Redis enabled: {db.redis is not None}")

    # Health check
    print("\n🏥 Running health check...")
    health = await db.health_check()

    print(f"\n📊 Health Status: {health.status.upper()}")
    print(f"\nService Status:")
    print(f"   Neo4j:  {health.neo4j.status.upper()} ({health.neo4j.latency_ms:.2f}ms)")
    print(f"   Qdrant: {health.qdrant.status.upper()} ({health.qdrant.latency_ms:.2f}ms)")
    print(f"   Redis:  {health.redis.status.upper()} ({health.redis.latency_ms:.2f}ms)")

    # Close database
    await db.close()
    print("\n✅ Database connections closed")


async def demo_role_based_access():
    """Demonstrate role-based access control"""
    print("\n" + "=" * 60)
    print("3. Role-Based Access Control Demo")
    print("=" * 60)

    # Create users with different roles
    roles_to_test = [
        (UserRole.STUDENT, "student_demo", "Student User"),
        (UserRole.TEACHER, "teacher_demo", "Teacher User"),
        (UserRole.PARENT, "parent_demo", "Parent User"),
        (UserRole.ADMIN, "admin_demo", "Admin User")
    ]

    print("\n📝 Creating users with different roles...")
    for role, username, full_name in roles_to_test:
        user = await create_user(
            username=username,
            email=f"{username}@edwin.edu",
            password="TestPassword123!@#",
            full_name=full_name,
            role=role
        )
        print(f"   ✅ {role.value:8s}: {username}")

    print("\n🔐 Testing role hierarchy...")
    print("   Role hierarchy: STUDENT < PARENT < TEACHER < ADMIN")

    # Test role comparison
    student_role = UserRole.STUDENT
    teacher_role = UserRole.TEACHER
    admin_role = UserRole.ADMIN

    print(f"   {student_role.value} < {teacher_role.value}: {student_role < teacher_role}")
    print(f"   {teacher_role.value} < {admin_role.value}: {teacher_role < admin_role}")


async def demo_production_features():
    """Demonstrate production-ready features"""
    print("\n" + "=" * 60)
    print("4. Production Features Demo")
    print("=" * 60)

    print("\n✅ Production features:")
    print("   🔒 JWT authentication with bcrypt password hashing")
    print("   🔑 Refresh token rotation")
    print("   👥 Role-based access control (RBAC)")
    print("   📊 Database health monitoring")
    print("   💾 Redis caching for sessions")
    print("   📈 Prometheus metrics export")
    print("   🚀 Kubernetes deployment manifests")
    print("   🐳 Docker containerization")
    print("   🔄 CI/CD with GitHub Actions")
    print("   📦 Horizontal Pod Autoscaling")
    print("   🛡️  Pod Disruption Budgets")
    print("   💽 Automated backups")

    print("\n✅ Security features:")
    print("   🔐 Password strength validation")
    print("   🔒 HTTPS/TLS enforcement")
    print("   🛡️  Rate limiting")
    print("   🔑 Session management")
    print("   📝 Audit logging")
    print("   🚫 CORS configuration")

    print("\n✅ Observability features:")
    print("   📊 Prometheus metrics")
    print("   📈 Grafana dashboards")
    print("   🔍 Structured logging")
    print("   🏥 Health check endpoints")
    print("   📉 Performance monitoring")


async def demo_deployment_workflow():
    """Show deployment workflow"""
    print("\n" + "=" * 60)
    print("5. Deployment Workflow")
    print("=" * 60)

    print("\n📋 Deployment Steps:")
    print("""
    1. Build Docker image:
       docker build -t edwin-api:latest .

    2. Start development environment:
       docker-compose -f docker-compose.edwin.yml up -d

    3. Initialize databases:
       ./scripts/init_db.sh

    4. Seed test data:
       python scripts/seed_data.py

    5. Run health check:
       ./scripts/health_check.sh

    6. Deploy to staging:
       ./scripts/deploy.sh staging

    7. Deploy to production (with approval):
       ./scripts/deploy.sh production

    8. Rollback if needed:
       ./scripts/rollback.sh production
    """)

    print("\n📋 Kubernetes Commands:")
    print("""
    # View pods
    kubectl get pods -n edwin

    # View services
    kubectl get svc -n edwin

    # View logs
    kubectl logs -f deployment/edwin-api -n edwin

    # Scale deployment
    kubectl scale deployment/edwin-api --replicas=5 -n edwin

    # Port forward for local access
    kubectl port-forward svc/edwin-api 8000:8000 -n edwin
    """)


async def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("EdWIN Production Deployment Demo")
    print("=" * 60)
    print("\nThis demo showcases EdWIN's production-ready features:")
    print("  - JWT authentication & authorization")
    print("  - Database connectivity & health monitoring")
    print("  - Role-based access control")
    print("  - Production deployment workflow")

    try:
        # Run demos
        await demo_authentication()
        await demo_database()
        await demo_role_based_access()
        await demo_production_features()
        await demo_deployment_workflow()

        print("\n" + "=" * 60)
        print("✅ Demo Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Review DEPLOYMENT.md for full deployment guide")
        print("  2. Configure production secrets in k8s/secrets.production.yaml")
        print("  3. Run ./scripts/deploy.sh staging to deploy to staging")
        print("  4. Test staging deployment thoroughly")
        print("  5. Deploy to production with approval workflow")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
