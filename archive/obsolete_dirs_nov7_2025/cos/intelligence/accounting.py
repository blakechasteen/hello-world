"""
HoloLoom COS - Core Accounting Documents

Generates the 4 fundamental accounting documents:
1. Income Statement (P&L)
2. Balance Sheet
3. Cash Flow Statement
4. Budget vs Actual

Philosophy: All derived from event stream, never stored separately.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio

from cos.core.types import Event, EventType, Category, ProductLine
from cos.core.storage import EventStore


@dataclass
class IncomeStatement:
    """
    Profit & Loss Statement

    Revenue - COGS - Operating Expenses = Net Profit
    """
    period_start: datetime
    period_end: datetime

    # Revenue by product line
    revenue_by_product: Dict[str, Decimal] = field(default_factory=dict)

    # Cost of Goods Sold
    cogs_materials: Decimal = Decimal('0')
    cogs_packaging: Decimal = Decimal('0')

    # Operating Expenses
    labor: Decimal = Decimal('0')
    overhead_utilities: Decimal = Decimal('0')
    overhead_rent: Decimal = Decimal('0')
    overhead_equipment: Decimal = Decimal('0')
    overhead_marketing: Decimal = Decimal('0')
    overhead_other: Decimal = Decimal('0')

    @property
    def total_revenue(self) -> Decimal:
        return sum(self.revenue_by_product.values())

    @property
    def total_cogs(self) -> Decimal:
        return self.cogs_materials + self.cogs_packaging

    @property
    def gross_profit(self) -> Decimal:
        return self.total_revenue - self.total_cogs

    @property
    def gross_margin(self) -> Decimal:
        if self.total_revenue == 0:
            return Decimal('0')
        return (self.gross_profit / self.total_revenue) * 100

    @property
    def total_operating_expenses(self) -> Decimal:
        return (self.labor + self.overhead_utilities + self.overhead_rent +
                self.overhead_equipment + self.overhead_marketing + self.overhead_other)

    @property
    def net_profit(self) -> Decimal:
        return self.gross_profit - self.total_operating_expenses

    @property
    def net_margin(self) -> Decimal:
        if self.total_revenue == 0:
            return Decimal('0')
        return (self.net_profit / self.total_revenue) * 100

    def to_text(self) -> str:
        """Format as text report"""
        period = f"{self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}"

        text = f"""
INCOME STATEMENT (P&L)
{period}
{'='*60}

REVENUE:
"""
        for product, revenue in sorted(self.revenue_by_product.items()):
            text += f"  {product:20} ${revenue:>12.2f}\n"
        text += f"  {'─'*35}\n"
        text += f"  {'TOTAL REVENUE':20} ${self.total_revenue:>12.2f}\n\n"

        text += f"""COST OF GOODS SOLD:
  Materials            ${self.cogs_materials:>12.2f}
  Packaging            ${self.cogs_packaging:>12.2f}
  {'─'*35}
  TOTAL COGS           ${self.total_cogs:>12.2f}

GROSS PROFIT           ${self.gross_profit:>12.2f}  ({self.gross_margin:>5.1f}%)

OPERATING EXPENSES:
  Labor                ${self.labor:>12.2f}
  Utilities            ${self.overhead_utilities:>12.2f}
  Rent                 ${self.overhead_rent:>12.2f}
  Equipment            ${self.overhead_equipment:>12.2f}
  Marketing            ${self.overhead_marketing:>12.2f}
  Other                ${self.overhead_other:>12.2f}
  {'─'*35}
  TOTAL OPEX           ${self.total_operating_expenses:>12.2f}

{'='*60}
NET PROFIT             ${self.net_profit:>12.2f}  ({self.net_margin:>5.1f}%)
{'='*60}
"""
        return text


@dataclass
class BalanceSheet:
    """
    Balance Sheet: Assets = Liabilities + Equity

    Snapshot of financial position at a point in time.
    """
    as_of_date: datetime

    # Assets
    cash: Decimal = Decimal('0')
    inventory_raw_materials: Decimal = Decimal('0')
    inventory_finished_goods: Decimal = Decimal('0')
    equipment: Decimal = Decimal('0')

    # Liabilities
    accounts_payable: Decimal = Decimal('0')
    loans: Decimal = Decimal('0')

    # Equity
    owner_investment: Decimal = Decimal('0')
    retained_earnings: Decimal = Decimal('0')

    @property
    def total_assets(self) -> Decimal:
        return (self.cash + self.inventory_raw_materials +
                self.inventory_finished_goods + self.equipment)

    @property
    def total_liabilities(self) -> Decimal:
        return self.accounts_payable + self.loans

    @property
    def total_equity(self) -> Decimal:
        return self.owner_investment + self.retained_earnings

    @property
    def is_balanced(self) -> bool:
        """Assets should equal Liabilities + Equity"""
        return abs(self.total_assets - (self.total_liabilities + self.total_equity)) < Decimal('0.01')

    def to_text(self) -> str:
        """Format as text report"""
        date = self.as_of_date.strftime('%Y-%m-%d')

        text = f"""
BALANCE SHEET
As of {date}
{'='*60}

ASSETS:
  Cash                 ${self.cash:>12.2f}
  Inventory (RM)       ${self.inventory_raw_materials:>12.2f}
  Inventory (FG)       ${self.inventory_finished_goods:>12.2f}
  Equipment            ${self.equipment:>12.2f}
  {'─'*35}
  TOTAL ASSETS         ${self.total_assets:>12.2f}

LIABILITIES:
  Accounts Payable     ${self.accounts_payable:>12.2f}
  Loans                ${self.loans:>12.2f}
  {'─'*35}
  TOTAL LIABILITIES    ${self.total_liabilities:>12.2f}

EQUITY:
  Owner Investment     ${self.owner_investment:>12.2f}
  Retained Earnings    ${self.retained_earnings:>12.2f}
  {'─'*35}
  TOTAL EQUITY         ${self.total_equity:>12.2f}

{'='*60}
TOTAL L + E            ${self.total_liabilities + self.total_equity:>12.2f}

Balance Check: {'✓ BALANCED' if self.is_balanced else '✗ NOT BALANCED'}
{'='*60}
"""
        return text


@dataclass
class CashFlowStatement:
    """
    Cash Flow Statement

    Tracks actual cash movement (not accrual accounting).
    """
    period_start: datetime
    period_end: datetime

    # Operating Activities
    cash_from_sales: Decimal = Decimal('0')
    cash_for_materials: Decimal = Decimal('0')
    cash_for_overhead: Decimal = Decimal('0')

    # Investing Activities
    cash_for_equipment: Decimal = Decimal('0')

    # Financing Activities
    owner_investment: Decimal = Decimal('0')
    owner_draws: Decimal = Decimal('0')
    loan_proceeds: Decimal = Decimal('0')
    loan_payments: Decimal = Decimal('0')

    # Opening/Closing Cash
    opening_cash: Decimal = Decimal('0')

    @property
    def operating_cash_flow(self) -> Decimal:
        return self.cash_from_sales - self.cash_for_materials - self.cash_for_overhead

    @property
    def investing_cash_flow(self) -> Decimal:
        return -self.cash_for_equipment

    @property
    def financing_cash_flow(self) -> Decimal:
        return (self.owner_investment - self.owner_draws +
                self.loan_proceeds - self.loan_payments)

    @property
    def net_change_in_cash(self) -> Decimal:
        return (self.operating_cash_flow + self.investing_cash_flow +
                self.financing_cash_flow)

    @property
    def closing_cash(self) -> Decimal:
        return self.opening_cash + self.net_change_in_cash

    def to_text(self) -> str:
        """Format as text report"""
        period = f"{self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}"

        text = f"""
CASH FLOW STATEMENT
{period}
{'='*60}

OPERATING ACTIVITIES:
  Cash from Sales      ${self.cash_from_sales:>12.2f}
  Cash for Materials   ${-self.cash_for_materials:>12.2f}
  Cash for Overhead    ${-self.cash_for_overhead:>12.2f}
  {'─'*35}
  Net Operating CF     ${self.operating_cash_flow:>12.2f}

INVESTING ACTIVITIES:
  Equipment Purchases  ${-self.cash_for_equipment:>12.2f}
  {'─'*35}
  Net Investing CF     ${self.investing_cash_flow:>12.2f}

FINANCING ACTIVITIES:
  Owner Investment     ${self.owner_investment:>12.2f}
  Owner Draws          ${-self.owner_draws:>12.2f}
  Loan Proceeds        ${self.loan_proceeds:>12.2f}
  Loan Payments        ${-self.loan_payments:>12.2f}
  {'─'*35}
  Net Financing CF     ${self.financing_cash_flow:>12.2f}

{'='*60}
NET CHANGE IN CASH     ${self.net_change_in_cash:>12.2f}

Opening Cash           ${self.opening_cash:>12.2f}
Closing Cash           ${self.closing_cash:>12.2f}
{'='*60}
"""
        return text


@dataclass
class BudgetVsActual:
    """
    Budget vs Actual Comparison

    Compares planned vs actual performance.
    """
    period_start: datetime
    period_end: datetime

    # Revenue
    budgeted_revenue: Decimal = Decimal('0')
    actual_revenue: Decimal = Decimal('0')

    # COGS
    budgeted_cogs: Decimal = Decimal('0')
    actual_cogs: Decimal = Decimal('0')

    # Labor
    budgeted_labor: Decimal = Decimal('0')
    actual_labor: Decimal = Decimal('0')

    # Overhead
    budgeted_overhead: Decimal = Decimal('0')
    actual_overhead: Decimal = Decimal('0')

    @property
    def budgeted_profit(self) -> Decimal:
        return (self.budgeted_revenue - self.budgeted_cogs -
                self.budgeted_labor - self.budgeted_overhead)

    @property
    def actual_profit(self) -> Decimal:
        return (self.actual_revenue - self.actual_cogs -
                self.actual_labor - self.actual_overhead)

    def variance(self, budgeted: Decimal, actual: Decimal) -> Decimal:
        return actual - budgeted

    def variance_pct(self, budgeted: Decimal, actual: Decimal) -> Decimal:
        if budgeted == 0:
            return Decimal('0')
        return (self.variance(budgeted, actual) / budgeted) * 100

    def to_text(self) -> str:
        """Format as text report"""
        period = f"{self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}"

        def format_line(label: str, budgeted: Decimal, actual: Decimal):
            var = self.variance(budgeted, actual)
            var_pct = self.variance_pct(budgeted, actual)
            status = "✓" if var >= 0 else "⚠"
            return f"  {label:15} ${budgeted:>10.2f}  ${actual:>10.2f}  {status} ${var:>9.2f}  ({var_pct:>6.1f}%)\n"

        text = f"""
BUDGET vs ACTUAL
{period}
{'='*80}

                    Budget        Actual      Variance        %
  {'─'*75}
"""
        text += format_line("Revenue", self.budgeted_revenue, self.actual_revenue)
        text += format_line("COGS", self.budgeted_cogs, self.actual_cogs)
        text += format_line("Labor", self.budgeted_labor, self.actual_labor)
        text += format_line("Overhead", self.budgeted_overhead, self.actual_overhead)
        text += f"  {'─'*75}\n"
        text += format_line("PROFIT", self.budgeted_profit, self.actual_profit)

        text += f"\n{'='*80}\n"
        return text


class AccountingGenerator:
    """
    Generates all 4 core accounting documents from event stream.
    """

    def __init__(self, store: EventStore):
        self.store = store

    async def generate_income_statement(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> IncomeStatement:
        """Generate P&L for period"""
        stmt = IncomeStatement(period_start=start_date, period_end=end_date)

        # Query all events in period
        events = await self.store.query(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )

        for event in events:
            if event.type == EventType.SALE:
                # Revenue
                product = event.product_line.value if event.product_line else 'other'
                stmt.revenue_by_product[product] = \
                    stmt.revenue_by_product.get(product, Decimal('0')) + event.amount

            elif event.type == EventType.PURCHASE:
                # COGS and Overhead
                if event.category == Category.COGS_MATERIALS:
                    stmt.cogs_materials += abs(event.amount)
                elif event.category == Category.COGS_PACKAGING:
                    stmt.cogs_packaging += abs(event.amount)
                elif event.category == Category.OVERHEAD_UTILITIES:
                    stmt.overhead_utilities += abs(event.amount)
                elif event.category == Category.OVERHEAD_RENT:
                    stmt.overhead_rent += abs(event.amount)
                elif event.category == Category.OVERHEAD_EQUIPMENT:
                    stmt.overhead_equipment += abs(event.amount)
                elif event.category == Category.OVERHEAD_MARKETING:
                    stmt.overhead_marketing += abs(event.amount)
                elif event.category == Category.OVERHEAD_OTHER:
                    stmt.overhead_other += abs(event.amount)

            elif event.type == EventType.TASK:
                # Labor
                stmt.labor += abs(event.amount)

        return stmt

    async def generate_balance_sheet(self, as_of_date: datetime) -> BalanceSheet:
        """Generate balance sheet as of date"""
        sheet = BalanceSheet(as_of_date=as_of_date)

        # Query ALL events up to date
        events = await self.store.query(
            end_date=as_of_date,
            limit=100000
        )

        # Calculate cash (cumulative)
        for event in events:
            if event.amount:
                if event.type in (EventType.SALE, EventType.PURCHASE):
                    sheet.cash += event.amount  # Sales +, purchases -
                elif event.category == Category.OWNER_INVESTMENT:
                    sheet.cash += event.amount
                    sheet.owner_investment += event.amount
                elif event.category == Category.OWNER_DRAW:
                    sheet.cash += event.amount  # Negative
                elif event.category == Category.CAPEX:
                    sheet.cash -= abs(event.amount)
                    sheet.equipment += abs(event.amount)

        # Calculate inventory (from inventory events)
        inventory = await self.store.get_inventory()
        # TODO: Separate raw materials vs finished goods
        # For now, lump all inventory as finished goods
        total_inventory_value = sum(
            item['quantity'] * item['avg_cost']
            for item in inventory.values()
        )
        sheet.inventory_finished_goods = Decimal(str(total_inventory_value))

        # Retained earnings = cumulative profit
        # Generate P&L from beginning of time to now
        pl = await self.generate_income_statement(
            datetime(2020, 1, 1),  # Beginning of time
            as_of_date
        )
        sheet.retained_earnings = pl.net_profit

        return sheet

    async def generate_cash_flow(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> CashFlowStatement:
        """Generate cash flow statement for period"""
        stmt = CashFlowStatement(period_start=start_date, period_end=end_date)

        # Get opening cash (balance sheet at start)
        opening_bs = await self.generate_balance_sheet(start_date - timedelta(days=1))
        stmt.opening_cash = opening_bs.cash

        # Query events in period
        events = await self.store.query(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )

        for event in events:
            if event.type == EventType.SALE:
                stmt.cash_from_sales += event.amount

            elif event.type == EventType.PURCHASE:
                if event.category in (Category.COGS_MATERIALS, Category.COGS_PACKAGING):
                    stmt.cash_for_materials += abs(event.amount)
                elif event.category.value.startswith('overhead'):
                    stmt.cash_for_overhead += abs(event.amount)
                elif event.category == Category.CAPEX:
                    stmt.cash_for_equipment += abs(event.amount)

            elif event.category == Category.OWNER_INVESTMENT:
                stmt.owner_investment += event.amount

            elif event.category == Category.OWNER_DRAW:
                stmt.owner_draws += abs(event.amount)

        return stmt

    async def generate_budget_vs_actual(
        self,
        start_date: datetime,
        end_date: datetime,
        budgeted_revenue: Decimal,
        budgeted_cogs: Decimal,
        budgeted_labor: Decimal,
        budgeted_overhead: Decimal
    ) -> BudgetVsActual:
        """Generate budget vs actual for period"""
        # Get actual from P&L
        pl = await self.generate_income_statement(start_date, end_date)

        return BudgetVsActual(
            period_start=start_date,
            period_end=end_date,
            budgeted_revenue=budgeted_revenue,
            actual_revenue=pl.total_revenue,
            budgeted_cogs=budgeted_cogs,
            actual_cogs=pl.total_cogs,
            budgeted_labor=budgeted_labor,
            actual_labor=pl.labor,
            budgeted_overhead=budgeted_overhead,
            actual_overhead=pl.total_operating_expenses - pl.labor
        )


async def demo():
    """Demo accounting document generation"""
    # Fix UTF-8 on Windows
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    print("\n" + "="*60)
    print("HoloLoom COS - Accounting Documents Demo")
    print("="*60)

    store = EventStore("test_cos.db")
    generator = AccountingGenerator(store)

    # Generate documents for this week
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    week_end = today

    # 1. Income Statement
    print("\n\n1. INCOME STATEMENT")
    pl = await generator.generate_income_statement(week_start, week_end)
    print(pl.to_text())

    # 2. Balance Sheet
    print("\n\n2. BALANCE SHEET")
    bs = await generator.generate_balance_sheet(today)
    print(bs.to_text())

    # 3. Cash Flow
    print("\n\n3. CASH FLOW STATEMENT")
    cf = await generator.generate_cash_flow(week_start, week_end)
    print(cf.to_text())

    # 4. Budget vs Actual (using Week 1 targets from 90-day plan)
    print("\n\n4. BUDGET vs ACTUAL")
    bva = await generator.generate_budget_vs_actual(
        week_start, week_end,
        budgeted_revenue=Decimal('200'),  # Week 1 target: $150-250
        budgeted_cogs=Decimal('30'),
        budgeted_labor=Decimal('60'),
        budgeted_overhead=Decimal('20')
    )
    print(bva.to_text())

    print("\n\n" + "="*60)
    print("✅ Accounting Demo Complete")
    print("="*60)


if __name__ == '__main__':
    asyncio.run(demo())
