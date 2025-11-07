"""
HoloLoom COS - Intelligence Layer

Business intelligence powered by HoloLoom agentic reasoning.

Modules:
- hololoom_integration: Event → Memory conversion, agentic queries
- accounting: 4 core accounting documents (P&L, Balance Sheet, Cash Flow, Budget)
- alerts: Business health monitoring and warnings
"""

from .hololoom_integration import COSMemoryBridge, COSAgenticInterface
from .accounting import (
    IncomeStatement, BalanceSheet, CashFlowStatement, BudgetVsActual,
    AccountingGenerator
)

__all__ = [
    'COSMemoryBridge',
    'COSAgenticInterface',
    'IncomeStatement',
    'BalanceSheet',
    'CashFlowStatement',
    'BudgetVsActual',
    'AccountingGenerator',
]
