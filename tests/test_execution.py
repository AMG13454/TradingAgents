"""Execution: a decision becomes a sized bracket order, or a stated reason for none.

The agents choose a direction; quantity and exits are arithmetic on the account.
The exits ride with the entry so the broker holds them, and the trading endpoint
is the paper one.
"""

from __future__ import annotations

import pytest

import tradingagents.execution as ex
from tradingagents.agents.schemas import (
    PortfolioDecision,
    TraderProposal,
    render_pm_decision,
    render_trader_proposal,
)
from tradingagents.execution import RiskLimits, execute_decision, plan_entry
from tradingagents.execution.alpaca import PAPER_URL, Account, AlpacaPaper, BrokerError
from tradingagents.execution.plan import BracketOrder, stated_levels

TODAY = "2026-11-13"


def _plan(rating="Buy", price=100.0, equity=100_000.0, buying_power=100_000.0, **kw):
    return plan_entry("NVDA", rating, price, equity, buying_power, **kw)


# --- sizing ------------------------------------------------------------------

@pytest.mark.unit
def test_quantity_is_the_risk_budget_over_the_stop_distance():
    # 0.5% of 100,000 is 500 at risk; a stop 5 below the price risks 5 a share.
    plan = _plan(stop=95.0, target=115.0)
    assert plan.order == BracketOrder("NVDA", 100, 95.0, 115.0)


@pytest.mark.unit
def test_an_overweight_is_sized_below_a_buy():
    assert _plan("Overweight", stop=95.0, target=115.0).order.qty == 50


@pytest.mark.unit
def test_a_tight_stop_cannot_size_past_the_position_cap():
    # A 2% stop would buy 250 shares on risk alone: 25% of equity in one name.
    plan = _plan(stop=98.0, target=110.0)
    assert plan.order.qty == 100 and any("capped at 10%" in n for n in plan.notes)


@pytest.mark.unit
def test_buying_power_caps_the_size():
    plan = _plan(buying_power=2_050.0, stop=95.0, target=115.0)
    assert plan.order.qty == 20 and any("buying power" in n for n in plan.notes)


@pytest.mark.unit
def test_a_budget_below_one_share_gives_no_order():
    plan = _plan(price=5_000.0, equity=1_000.0, buying_power=1_000.0)
    assert plan.order is None and "whole shares" in plan.render()


# --- exits -------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("stop", [None, 105.0, 99.5, 60.0, float("nan")])
def test_a_missing_or_unusable_stop_becomes_the_default_distance(stop):
    """A stop above the price, within noise of it, or too far to protect is the
    model's error; the order still gets a stop, by rule."""
    assert _plan(stop=stop, target=130.0).order.stop_price == 92.0


@pytest.mark.unit
@pytest.mark.parametrize("target", [None, 90.0, 102.0])
def test_a_target_that_pays_less_than_the_risk_becomes_a_multiple_of_it(target):
    assert _plan(stop=95.0, target=target).order.take_profit_price == 110.0


@pytest.mark.unit
def test_every_order_has_both_exits_on_the_right_side_of_the_price():
    order = _plan().order
    assert order.stop_price < 100.0 < order.take_profit_price


# --- when there is no order --------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("rating", ["Hold", "Underweight", "Sell", "REVIEW"])
def test_only_a_bullish_rating_opens_a_position(rating):
    plan = _plan(rating)
    assert plan.order is None and rating in plan.notes[0]


@pytest.mark.unit
def test_a_held_symbol_is_not_added_to():
    assert _plan(held_qty=10).order is None


@pytest.mark.unit
def test_a_symbol_with_an_open_order_is_not_entered_twice():
    assert _plan(open_orders=1).order is None


@pytest.mark.unit
@pytest.mark.parametrize("kw", [{"risk_per_trade": 0.2}, {"risk_per_trade": 0}, {"max_position": 1.5},
                                {"min_stop": 0.1, "default_stop": 0.05}])
def test_limits_that_would_bet_the_account_are_rejected(kw):
    with pytest.raises(ValueError):
        RiskLimits(**kw)


# --- levels from the run -----------------------------------------------------

def _state(rating="Buy", action="Buy", stop=95.0, target=115.0, ticker="NVDA", date=TODAY):
    trader = render_trader_proposal(TraderProposal(action=action, reasoning="r", stop_loss=stop))
    decision = render_pm_decision(PortfolioDecision(
        rating=rating, executive_summary="s", investment_thesis="t", price_target=target))
    return {"company_of_interest": ticker, "trade_date": date, "asset_type": "stock",
            "trader_investment_plan": trader, "final_trade_decision": decision, "final_rating": rating}


@pytest.mark.unit
def test_levels_are_read_from_the_lines_the_renderers_write():
    assert stated_levels(_state()) == (95.0, 115.0)


@pytest.mark.unit
def test_levels_not_given_are_absent():
    assert stated_levels(_state(stop=None, target=None)) == (None, None)


@pytest.mark.unit
def test_the_stop_of_a_sell_proposal_is_not_an_entry_stop():
    assert stated_levels(_state(action="Sell", stop=108.0))[0] is None


@pytest.mark.unit
def test_a_free_text_decision_has_no_levels_rather_than_guessed_ones():
    state = {"trader_investment_plan": "I would buy with a stop loss near 95.",
             "final_trade_decision": "Rating: Buy\n\nTarget is about 115."}
    assert stated_levels(state) == (None, None)


# --- the whole path ----------------------------------------------------------

class _Broker:
    def __init__(self, is_open=True, blocked=False, held=0.0, orders=0, price=100.0):
        self.is_open, self.blocked, self.held, self.orders, self.price = is_open, blocked, held, orders, price
        self.submitted = []

    def account(self):
        return Account(equity=100_000.0, buying_power=100_000.0, blocked=self.blocked)

    def market_is_open(self):
        return self.is_open

    def held_qty(self, symbol):
        return self.held

    def open_orders(self, symbol):
        return self.orders

    def last_price(self, symbol):
        return self.price

    def submit_bracket(self, order, client_order_id):
        self.submitted.append((order, client_order_id))
        return {"id": "abc-123"}


@pytest.fixture(autouse=True)
def _today(monkeypatch):
    monkeypatch.setattr(ex, "get_current_date", lambda: TODAY)


@pytest.mark.unit
def test_nothing_is_sent_unless_asked():
    broker = _Broker()
    report = execute_decision(_state(), broker)
    assert report.plan.order is not None and not report.submitted and broker.submitted == []
    assert "dry run" in report.render()


@pytest.mark.unit
def test_a_submitted_order_carries_one_id_per_symbol_and_date():
    """Re-running the same day's decision reuses the id, and the broker refuses
    an id it has seen, so a retry cannot double the position."""
    broker = _Broker()
    report = execute_decision(_state(), broker, submit=True)
    order, client_id = broker.submitted[0]
    assert order == BracketOrder("NVDA", 100, 95.0, 115.0)
    assert client_id == f"tradingagents-NVDA-{TODAY}"
    assert report.submitted and "abc-123" in report.render()


@pytest.mark.unit
def test_a_closed_market_takes_no_order():
    broker = _Broker(is_open=False)
    report = execute_decision(_state(), broker, submit=True)
    assert report.plan.order is None and "closed" in report.render() and broker.submitted == []


@pytest.mark.unit
def test_a_blocked_account_takes_no_order():
    assert execute_decision(_state(), _Broker(blocked=True), submit=True).plan.order is None


@pytest.mark.unit
def test_a_decision_from_another_day_is_not_traded():
    broker = _Broker()
    report = execute_decision(_state(date="2026-11-01"), broker, submit=True)
    assert report.plan.order is None and "2026-11-01" in report.render() and broker.submitted == []


@pytest.mark.unit
@pytest.mark.parametrize("ticker", ["0700.HK", "BTC-USD", "GC=F", "7203.T"])
def test_an_instrument_the_broker_does_not_serve_is_refused_before_any_call(ticker):
    report = execute_decision(_state(ticker=ticker), broker=None, submit=True)
    assert report.plan.order is None and "US-listed" in report.render()


@pytest.mark.unit
def test_a_bearish_decision_on_a_live_market_sends_nothing():
    broker = _Broker()
    assert execute_decision(_state(rating="Sell", action="Sell"), broker, submit=True).plan.order is None
    assert broker.submitted == []


# --- the broker client -------------------------------------------------------

class _Response:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, str(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.replies[(method, url)]


def _client(replies):
    session = _Session(replies)
    return AlpacaPaper("key", "secret", session=session), session


@pytest.mark.unit
def test_the_exits_ride_with_the_entry_and_outlive_the_day():
    """A day order's exits lapse at the close, leaving the position with neither."""
    client, session = _client({("POST", f"{PAPER_URL}/v2/orders"): _Response(200, {"id": "1"})})
    client.submit_bracket(BracketOrder("NVDA", 100, 95.0, 115.0), "id-1")
    body = session.calls[0][2]["json"]
    assert body["order_class"] == "bracket" and body["time_in_force"] == "gtc"
    assert body["stop_loss"] == {"stop_price": "95.00"}
    assert body["take_profit"] == {"limit_price": "115.00"}
    assert body["side"] == "buy" and body["qty"] == "100" and body["client_order_id"] == "id-1"


@pytest.mark.unit
def test_orders_go_to_the_paper_endpoint_only():
    assert PAPER_URL == "https://paper-api.alpaca.markets"
    client, session = _client({("POST", f"{PAPER_URL}/v2/orders"): _Response(200, {"id": "1"})})
    client.submit_bracket(BracketOrder("NVDA", 1, 95.0, 115.0), "id-1")
    assert all(url.startswith(PAPER_URL) for _, url, _ in session.calls)


@pytest.mark.unit
def test_no_position_reads_as_zero_shares():
    client, _ = _client({("GET", f"{PAPER_URL}/v2/positions/NVDA"): _Response(404, {"message": "none"})})
    assert client.held_qty("NVDA") == 0.0


@pytest.mark.unit
def test_a_refusal_names_the_brokers_reason():
    client, _ = _client({("POST", f"{PAPER_URL}/v2/orders"):
                         _Response(422, {"message": "client_order_id must be unique"})})
    with pytest.raises(BrokerError, match="must be unique"):
        client.submit_bracket(BracketOrder("NVDA", 1, 95.0, 115.0), "id-1")


@pytest.mark.unit
def test_missing_keys_are_reported_before_any_request(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(BrokerError, match="APCA_API_KEY_ID"):
        AlpacaPaper()
