#!/usr/bin/env python3
"""
HoloLoom SaaS Toolkit Examples

Example applications demonstrating different usage patterns:

1. auth_only_app.py
   - Minimal authentication without billing
   - SQLite backend, no external dependencies
   - Run: PYTHONPATH=. uvicorn HoloLoom.saas.examples.auth_only_app:app --reload

2. usage_tracking_app.py
   - Authentication + usage analytics (no billing)
   - Tracks queries, tokens, and events
   - Run: PYTHONPATH=. uvicorn HoloLoom.saas.examples.usage_tracking_app:app --reload

3. full_billing_app.py
   - Complete Stripe billing integration
   - Subscription management, webhooks, overage billing
   - Run: STRIPE_API_KEY=sk_... PYTHONPATH=. uvicorn HoloLoom.saas.examples.full_billing_app:app --reload

Each example demonstrates progressive complexity:
- Start with auth_only_app for development
- Add usage_tracking when you need analytics
- Use full_billing when you need monetization
"""
