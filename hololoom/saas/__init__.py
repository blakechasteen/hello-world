#!/usr/bin/env python3
"""
HoloLoom SaaS Package

Phase 1: API-only SaaS with usage-based billing
- Customers bring their own backend (BYO)
- Usage-based pricing (per-query)
- Stripe integration for billing

Components:
- backend.py: Database CRUD operations
- models.py: Pydantic request/response models
- auth.py: API key validation middleware
- metering.py: Usage tracking middleware
- stripe_client.py: Stripe API wrapper
- routes/: FastAPI endpoint routers
- jobs/: Background jobs (usage reporting)
"""

from .backend import (
    SaaSBackend,
    SaaSConfig,
    Customer,
    Subscription,
    APIKey,
    UsageRecord,
    QueryResult,
    create_saas_backend,
)

from .models import (
    # Customer models
    SignupRequest,
    SignupResponse,
    LoginRequest,
    LoginResponse,
    CustomerProfile,
    CustomerUpdateRequest,
    # API Key models
    APIKeyCreateRequest,
    APIKeyResponse,
    APIKeyCreateResponse,
    APIKeyListResponse,
    # Subscription models
    SubscriptionResponse,
    UpgradeRequest,
    UpgradeResponse,
    CancelResponse,
    # Usage models
    UsageDay,
    UsageSummary,
    UsageResponse,
    # Billing models
    Invoice,
    InvoiceListResponse,
    PaymentMethod,
    BillingInfo,
    # Error models
    ErrorResponse,
    RateLimitError,
    AuthError,
    ValidationError,
    # Plan configuration
    PlanConfig,
    PLAN_CONFIGS,
    get_plan_config,
)

__all__ = [
    # Backend
    "SaaSBackend",
    "SaaSConfig",
    "Customer",
    "Subscription",
    "APIKey",
    "UsageRecord",
    "QueryResult",
    "create_saas_backend",
    # Customer models
    "SignupRequest",
    "SignupResponse",
    "LoginRequest",
    "LoginResponse",
    "CustomerProfile",
    "CustomerUpdateRequest",
    # API Key models
    "APIKeyCreateRequest",
    "APIKeyResponse",
    "APIKeyCreateResponse",
    "APIKeyListResponse",
    # Subscription models
    "SubscriptionResponse",
    "UpgradeRequest",
    "UpgradeResponse",
    "CancelResponse",
    # Usage models
    "UsageDay",
    "UsageSummary",
    "UsageResponse",
    # Billing models
    "Invoice",
    "InvoiceListResponse",
    "PaymentMethod",
    "BillingInfo",
    # Error models
    "ErrorResponse",
    "RateLimitError",
    "AuthError",
    "ValidationError",
    # Plan configuration
    "PlanConfig",
    "PLAN_CONFIGS",
    "get_plan_config",
]

__version__ = "1.0.0"
