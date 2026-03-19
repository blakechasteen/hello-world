#!/usr/bin/env python3
"""
Pydantic Models for HoloLoom SaaS API

Request/Response models for:
- Customer signup and authentication
- API key management
- Subscription management
- Usage tracking and dashboard
- Billing webhooks
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field, validator

# ============================================================================
# Customer Models
# ============================================================================

class SignupRequest(BaseModel):
    """Customer signup request"""
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=255)
    company: str | None = Field(None, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)

    @validator("name")
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()

    @validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        # Optional: Add complexity requirements
        # if not re.search(r"[A-Z]", v):
        #     raise ValueError("Password must contain an uppercase letter")
        # if not re.search(r"[a-z]", v):
        #     raise ValueError("Password must contain a lowercase letter")
        # if not re.search(r"\d", v):
        #     raise ValueError("Password must contain a digit")
        return v


class SignupResponse(BaseModel):
    """Customer signup response"""
    customer_id: str
    email: str
    name: str
    plan: str = "free"
    api_key_id: str
    api_secret: str = Field(..., description="API secret - shown ONLY on signup")
    message: str = "Save your API secret - it won't be shown again"


class LoginRequest(BaseModel):
    """Customer login request"""
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    """Customer login response"""
    customer_id: str
    email: str
    name: str
    plan: str
    access_token: str  # JWT token for session auth


class CustomerProfile(BaseModel):
    """Customer profile data"""
    customer_id: str
    email: str
    name: str
    company: str | None = None
    plan: str
    status: str
    created_at: datetime | None = None
    usage_this_month: int = 0
    queries_included: int = 0


class CustomerUpdateRequest(BaseModel):
    """Customer profile update request"""
    name: str | None = Field(None, min_length=1, max_length=255)
    company: str | None = Field(None, max_length=255)


# ============================================================================
# API Key Models
# ============================================================================

class APIKeyCreateRequest(BaseModel):
    """API key creation request"""
    name: str = Field("Default", min_length=1, max_length=255)
    expires_in_days: int | None = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response (no secret)"""
    key_id: str
    name: str
    key_prefix: str = Field(..., description="First 8 characters for identification")
    status: str
    rate_limit_qps: float
    created_at: datetime | None = None
    expires_at: datetime | None = None
    last_used: datetime | None = None
    request_count: int = 0


class APIKeyCreateResponse(BaseModel):
    """API key creation response (with secret)"""
    key_id: str
    name: str
    secret: str = Field(..., description="API secret - shown ONLY on creation")
    key_prefix: str
    rate_limit_qps: float
    expires_at: datetime | None = None
    message: str = "Save your API secret - it won't be shown again"


class APIKeyListResponse(BaseModel):
    """List of API keys"""
    api_keys: list[APIKeyResponse]


# ============================================================================
# Subscription Models
# ============================================================================

class SubscriptionResponse(BaseModel):
    """Subscription details"""
    subscription_id: str
    plan: str
    status: str
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    queries_included: int
    price_per_query_cents: float | None = None
    cancel_at_period_end: bool = False


class UpgradeRequest(BaseModel):
    """Plan upgrade request"""
    plan: str = Field(..., pattern="^(payg|pro|enterprise)$")


class UpgradeResponse(BaseModel):
    """Plan upgrade response"""
    success: bool
    checkout_url: str | None = None  # Stripe checkout URL
    new_plan: str
    effective_date: datetime | None = None
    message: str


class CancelResponse(BaseModel):
    """Subscription cancellation response"""
    success: bool
    cancels_at: datetime | None = None
    message: str


class PaymentMethodResponse(BaseModel):
    """Payment method response"""
    setup_url: str  # Stripe setup URL for updating payment method


# ============================================================================
# Usage Models
# ============================================================================

class UsageDay(BaseModel):
    """Daily usage data"""
    date: str  # ISO date string
    queries: int
    tokens: int = 0


class UsageSummary(BaseModel):
    """Usage summary for dashboard"""
    current_period: dict[str, str]  # {"start": ISO, "end": ISO}
    queries_used: int
    queries_included: int
    overage_queries: int
    estimated_cost_cents: int
    daily_breakdown: list[UsageDay]


class UsageResponse(BaseModel):
    """Usage dashboard response"""
    summary: UsageSummary
    plan: str
    rate_limit_qps: float


# ============================================================================
# Billing Models
# ============================================================================

class Invoice(BaseModel):
    """Invoice data"""
    id: str
    amount_cents: int
    status: str  # paid, open, void, uncollectible
    period_start: str
    period_end: str
    pdf_url: str | None = None
    created_at: str


class InvoiceListResponse(BaseModel):
    """List of invoices"""
    invoices: list[Invoice]


class PaymentMethod(BaseModel):
    """Payment method on file"""
    last4: str
    brand: str  # visa, mastercard, amex, etc.
    exp_month: int
    exp_year: int


class BillingInfo(BaseModel):
    """Billing information"""
    payment_method: PaymentMethod | None = None
    next_invoice_date: str | None = None
    next_invoice_amount_cents: int | None = None


# ============================================================================
# Webhook Models
# ============================================================================

class StripeWebhookPayload(BaseModel):
    """Stripe webhook event payload"""
    id: str
    type: str
    data: dict[str, Any]
    created: int


# ============================================================================
# Error Models
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    code: str
    details: dict[str, Any] | None = None


class RateLimitError(BaseModel):
    """Rate limit exceeded error"""
    error: str = "Rate limit exceeded"
    code: str = "rate_limit_exceeded"
    retry_after_seconds: int
    limit: float
    remaining: int = 0


class AuthError(BaseModel):
    """Authentication error"""
    error: str
    code: str = "authentication_failed"


class ValidationError(BaseModel):
    """Validation error"""
    error: str = "Validation failed"
    code: str = "validation_error"
    details: dict[str, list[str]]  # field -> list of errors


# ============================================================================
# Response Headers Models (for documentation)
# ============================================================================

class UsageHeaders(BaseModel):
    """Usage headers included in responses"""
    x_usage_queries: int = Field(..., alias="X-Usage-Queries")
    x_usage_limit: int | None = Field(None, alias="X-Usage-Limit")
    x_usage_remaining: int | None = Field(None, alias="X-Usage-Remaining")

    class Config:
        populate_by_name = True


class RateLimitHeaders(BaseModel):
    """Rate limit headers included in responses"""
    x_ratelimit_limit: float = Field(..., alias="X-RateLimit-Limit")
    x_ratelimit_remaining: int = Field(..., alias="X-RateLimit-Remaining")
    x_ratelimit_reset: int = Field(..., alias="X-RateLimit-Reset")

    class Config:
        populate_by_name = True


# ============================================================================
# Plan Configuration (Reference)
# ============================================================================

class PlanConfig(BaseModel):
    """Plan configuration"""
    plan_id: str
    name: str
    description: str
    queries_included: int
    price_per_query_cents: float
    rate_limit_qps: float
    features: dict[str, Any]


# Default plan configurations
PLAN_CONFIGS = {
    "free": PlanConfig(
        plan_id="free",
        name="Free",
        description="Free tier with 100 queries/day limit",
        queries_included=0,
        price_per_query_cents=0.0,
        rate_limit_qps=1.0,
        features={"daily_limit": 100, "support": "community"}
    ),
    "payg": PlanConfig(
        plan_id="payg",
        name="Pay As You Go",
        description="Usage-based pricing with no commitment",
        queries_included=0,
        price_per_query_cents=0.1,
        rate_limit_qps=10.0,
        features={"daily_limit": None, "support": "email"}
    ),
    "pro": PlanConfig(
        plan_id="pro",
        name="Pro",
        description="Professional tier with 50K included queries",
        queries_included=50000,
        price_per_query_cents=0.05,
        rate_limit_qps=50.0,
        features={"daily_limit": None, "support": "priority", "analytics": True}
    ),
    "enterprise": PlanConfig(
        plan_id="enterprise",
        name="Enterprise",
        description="Custom enterprise pricing",
        queries_included=0,
        price_per_query_cents=0.0,
        rate_limit_qps=100.0,
        features={"daily_limit": None, "support": "dedicated", "sla": True, "custom_pricing": True}
    )
}


def get_plan_config(plan: str) -> PlanConfig:
    """Get configuration for a plan"""
    return PLAN_CONFIGS.get(plan, PLAN_CONFIGS["free"])
