"""Acting on a decision: a sized bracket order on a paper account.

``execute_decision`` is the whole path, and it only plans unless told to submit:

    final_state, _ = graph.propagate("NVDA", today)
    report = execute_decision(final_state, AlpacaPaper(), submit=True)
    print(report.render())
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tradingagents.dataflows.date_window import get_current_date
from tradingagents.execution.alpaca import AlpacaPaper, BrokerError
from tradingagents.execution.plan import (
    BracketOrder,
    OrderPlan,
    RiskLimits,
    decision_rating,
    plan_entry,
    stated_levels,
)

__all__ = ["AlpacaPaper", "BracketOrder", "BrokerError", "ExecutionReport", "OrderPlan",
           "RiskLimits", "execute_decision", "plan_entry"]

# A US-listed share symbol, with a class letter when it has one (BRK.B). A
# suffixed foreign listing, a coin pair or a future is not served by the broker.
_US_SYMBOL = re.compile(r"[A-Z]{1,5}(\.[A-Z])?")


@dataclass
class ExecutionReport:
    plan: OrderPlan
    submitted: bool = False
    order_id: str | None = None

    def render(self) -> str:
        if self.plan.order is None:
            status = ""
        elif self.submitted:
            status = f"\nSubmitted to the paper account as order {self.order_id}."
        else:
            status = "\nNot submitted: this was a dry run."
        return self.plan.render() + status


def execute_decision(final_state: dict, broker, limits: RiskLimits | None = None,
                     submit: bool = False) -> ExecutionReport:
    """Plan the order a finished run's decision comes to, and submit it when asked.

    Every guard here ends in a plan with no order and a note saying why, so a
    scheduled caller can log the report whatever happened. A broker failure is
    raised as ``BrokerError``: an order whose fate is unknown is not a report.
    """
    symbol = str(final_state["company_of_interest"]).strip().upper()
    trade_date = final_state.get("trade_date")

    def no_order(reason: str) -> ExecutionReport:
        return ExecutionReport(OrderPlan(notes=[reason]))

    if final_state.get("asset_type", "stock") != "stock" or not _US_SYMBOL.fullmatch(symbol):
        return no_order(f"{symbol} is not a US-listed share: the paper account takes those only.")
    # A decision made on another day's data is not one to trade today.
    if trade_date != get_current_date():
        return no_order(f"The decision is dated {trade_date}, not today: only today's decision is traded.")

    account = broker.account()
    if account.blocked:
        return no_order("The account is blocked from trading.")
    # A market order sent while the market is closed fills at the next open,
    # at a price the stop and target were not measured against.
    if not broker.market_is_open():
        return no_order("The market is closed: an entry is sized against a live price.")

    stop, target = stated_levels(final_state)
    plan = plan_entry(
        symbol, decision_rating(final_state),
        price=broker.last_price(symbol),
        equity=account.equity, buying_power=account.buying_power,
        held_qty=broker.held_qty(symbol), open_orders=broker.open_orders(symbol),
        stop=stop, target=target, limits=limits,
    )
    report = ExecutionReport(plan)
    if plan.order is not None and submit:
        # One order per symbol per decision date, however often this is re-run.
        placed = broker.submit_bracket(plan.order, client_order_id=f"tradingagents-{symbol}-{trade_date}")
        report.submitted, report.order_id = True, placed.get("id")
    return report
