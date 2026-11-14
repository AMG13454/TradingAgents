"""Turning a decision into an order: quantity and exit levels, by rule.

The agents decide direction. They do not decide how much: a model's "5% of
portfolio" is prose, not a risk budget, and a stop it states can sit on the wrong
side of the price. Everything here is arithmetic on numbers the broker reports,
so the same decision and the same account always give the same order.

The order is a bracket: the entry carries its stop-loss and take-profit, and the
broker holds both from the fill onward. The exits then do not depend on this
process, this machine or its connection staying up.

Scope: long entries into a flat symbol. Ratings that would reduce, exit or short
produce no order; the reason is reported instead. Sizing is fixed-fractional,
the loss at the stop as a share of equity, because it needs no estimate of win
probability or payoff. A Kelly fraction needs both, and until a backtest has
measured them any value put there is invented.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from tradingagents.agents.rating import run_rating


@dataclass(frozen=True)
class RiskLimits:
    """The risk budget an entry is sized against. Fractions, not percentages."""

    risk_per_trade: float = 0.005     # equity lost if the stop fills at its price
    max_position: float = 0.10        # cap on the position's value, as a share of equity
    overweight_scale: float = 0.5     # an Overweight is sized at this share of a Buy
    min_stop: float = 0.02            # a stop closer than this to the price is noise
    max_stop: float = 0.20            # a stop farther than this protects nothing
    default_stop: float = 0.08        # the distance used when no usable stop was given
    reward_multiple: float = 2.0      # take-profit, in stop distances, when no usable target was given

    def __post_init__(self):
        if not 0 < self.risk_per_trade <= 0.05:
            raise ValueError("risk_per_trade must be above 0 and at most 0.05 of equity")
        if not 0 < self.max_position <= 1:
            raise ValueError("max_position must be above 0 and at most 1")
        if not 0 < self.overweight_scale <= 1:
            raise ValueError("overweight_scale must be above 0 and at most 1")
        if not 0 < self.min_stop <= self.default_stop <= self.max_stop < 1:
            raise ValueError("stop distances must satisfy 0 < min_stop <= default_stop <= max_stop < 1")
        if self.reward_multiple <= 0:
            raise ValueError("reward_multiple must be above 0")


@dataclass(frozen=True)
class BracketOrder:
    """A market buy with its two exits, in whole shares and prices to the cent."""

    symbol: str
    qty: int
    stop_price: float
    take_profit_price: float


@dataclass
class OrderPlan:
    """What a decision comes to: an order, or the reason there is none.

    ``notes`` records every choice the rules made, such as a stop that was
    replaced or a size that was capped, so the order can be audited afterwards.
    """

    order: BracketOrder | None = None
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        if self.order is None:
            head = "No order."
        else:
            o = self.order
            head = (f"Buy {o.qty} {o.symbol} at market, stop-loss {o.stop_price:.2f}, "
                    f"take-profit {o.take_profit_price:.2f}.")
        return "\n".join([head, *(f"- {note}" for note in self.notes)])


_ENTRY_SCALE = {"Buy": 1.0, "Overweight": None}   # None: the limits' overweight_scale


def plan_entry(
    symbol: str,
    rating: str,
    price: float,
    equity: float,
    buying_power: float,
    held_qty: float = 0,
    open_orders: int = 0,
    stop: float | None = None,
    target: float | None = None,
    limits: RiskLimits | None = None,
) -> OrderPlan:
    """Size a long bracket entry for ``rating``, or say why there is no order.

    ``price`` is the latest trade, ``equity`` and ``buying_power`` the account's,
    ``held_qty`` and ``open_orders`` what the account already has in the symbol.
    ``stop`` and ``target`` are the levels the agents stated, used only when they
    pass the limits' checks.
    """
    limits = limits or RiskLimits()
    plan = OrderPlan()
    if rating not in _ENTRY_SCALE:
        plan.notes.append(
            f"The rating is {rating}: only Buy and Overweight open a position. "
            "Reducing or closing one is left to its bracket exits and to you."
        )
        return plan
    if not all(math.isfinite(v) and v > 0 for v in (price, equity)):
        plan.notes.append(f"No usable price or equity (price {price}, equity {equity}).")
        return plan
    if held_qty:
        plan.notes.append(f"The account already holds {held_qty:g} {symbol}: an entry would add to it unsized.")
        return plan
    if open_orders:
        plan.notes.append(f"{symbol} has {open_orders} open order(s): a second entry would stack on the first.")
        return plan

    stop_price = _usable_stop(stop, price, limits, plan.notes)
    risk_per_share = price - stop_price
    take_profit = _usable_target(target, price, risk_per_share, limits, plan.notes)

    scale = _ENTRY_SCALE[rating] or limits.overweight_scale
    if scale != 1.0:
        plan.notes.append(f"{rating} is sized at {scale:.0%} of a Buy.")
    by_risk = equity * limits.risk_per_trade * scale / risk_per_share
    by_cap = equity * limits.max_position / price
    by_cash = max(buying_power, 0) / price
    qty = math.floor(min(by_risk, by_cap, by_cash))
    if by_cap < by_risk and by_cap <= by_cash:
        plan.notes.append(f"Size capped at {limits.max_position:.0%} of equity.")
    elif by_cash < by_risk:
        plan.notes.append("Size capped by buying power.")
    if qty < 1:
        plan.notes.append(
            f"The risk budget buys less than one share at {price:.2f} "
            "(a bracket order takes whole shares only)."
        )
        return plan

    plan.order = BracketOrder(symbol, qty, round(stop_price, 2), round(take_profit, 2))
    plan.notes.append(
        f"Loss if the stop fills at its price: {qty * risk_per_share:,.2f} "
        f"({qty * risk_per_share / equity:.2%} of equity). A gap through the stop loses more."
    )
    return plan


def _usable_stop(stop: float | None, price: float, limits: RiskLimits, notes: list[str]) -> float:
    """The stated stop when it sits within the limits below the price, else the default distance."""
    fallback = price * (1 - limits.default_stop)
    if stop is None:
        notes.append(f"No stop was stated: using {limits.default_stop:.0%} below the price.")
        return fallback
    distance = (price - stop) / price
    if not math.isfinite(stop) or not limits.min_stop <= distance <= limits.max_stop:
        notes.append(
            f"The stated stop {stop:g} is {distance:+.1%} from the price, outside "
            f"{limits.min_stop:.0%} to {limits.max_stop:.0%} below it: "
            f"using {limits.default_stop:.0%} below instead."
        )
        return fallback
    return stop


def _usable_target(target: float | None, price: float, risk_per_share: float,
                   limits: RiskLimits, notes: list[str]) -> float:
    """The stated target when it pays at least the risk taken, else a multiple of the stop distance."""
    fallback = price + limits.reward_multiple * risk_per_share
    if target is None:
        notes.append(f"No price target was stated: using {limits.reward_multiple:g}x the stop distance.")
        return fallback
    if not math.isfinite(target) or target < price + risk_per_share:
        notes.append(
            f"The stated target {target:g} pays less than the stop risks: "
            f"using {limits.reward_multiple:g}x the stop distance instead."
        )
        return fallback
    return target


# The Trader's and Portfolio Manager's renderers write these lines themselves
# (schemas.render_trader_proposal, render_pm_decision). A value read here is
# still checked against the price before it is used, so a line a free-text
# fallback wrote differently, or wrote wrongly, costs the level and not the order.
_STOP_LINE = re.compile(r"^\*\*Stop Loss\*\*:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
_TARGET_LINE = re.compile(r"^\*\*Price Target\*\*:\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)
_TRADER_BUY = re.compile(r"^\*\*Action\*\*:\s*Buy\s*$", re.MULTILINE)


def stated_levels(final_state: dict) -> tuple[float | None, float | None]:
    """The stop the Trader stated and the target the Portfolio Manager stated, when each gave one.

    The Trader's stop belongs to a Buy; the stop of a Sell proposal protects a
    different trade, so it is not carried over to an entry.
    """
    trader = final_state.get("trader_investment_plan") or ""
    stop = _STOP_LINE.search(trader) if _TRADER_BUY.search(trader) else None
    target = _TARGET_LINE.search(final_state.get("final_trade_decision") or "")
    return (float(stop.group(1)) if stop else None,
            float(target.group(1)) if target else None)


def decision_rating(final_state: dict) -> str:
    """The run's final rating, as the graph recorded it."""
    return run_rating(final_state)
