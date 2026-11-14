"""Alpaca paper trading: the account, the market clock and bracket orders.

Paper only. The trading endpoint is the paper one and is not configurable: this
system has no measured edge, and a path to a funded account would let it trade
on one anyway. Live keys do not authenticate against the paper endpoint.

Keys are read from ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY``, the names
Alpaca's own tools use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from tradingagents.execution.plan import BracketOrder

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
_TIMEOUT = 15


class BrokerError(RuntimeError):
    """The broker refused a request or could not be reached."""


@dataclass(frozen=True)
class Account:
    equity: float
    buying_power: float
    blocked: bool


class AlpacaPaper:
    """The calls an entry needs against an Alpaca paper account."""

    def __init__(self, key_id: str | None = None, secret_key: str | None = None, session=None):
        key_id = key_id or os.environ.get("APCA_API_KEY_ID")
        secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY")
        if not key_id or not secret_key:
            raise BrokerError(
                "Alpaca paper keys are not set: export APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY from your paper account."
            )
        self._session = session or requests.Session()
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret_key}

    def _call(self, method: str, url: str, missing_ok: bool = False, **kwargs):
        try:
            response = self._session.request(method, url, headers=self._headers,
                                             timeout=_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            raise BrokerError(f"Alpaca could not be reached: {exc}") from exc
        if missing_ok and response.status_code == 404:
            return None
        if response.status_code >= 400:
            # The body names the reason (a duplicate order id, a rejected price).
            raise BrokerError(f"Alpaca refused {method} {url.split('/v2/')[-1]}: "
                              f"{response.status_code} {response.text[:300]}")
        return response.json()

    def account(self) -> Account:
        data = self._call("GET", f"{PAPER_URL}/v2/account")
        return Account(
            equity=float(data["equity"]),
            buying_power=float(data["buying_power"]),
            blocked=bool(data.get("trading_blocked") or data.get("account_blocked")),
        )

    def market_is_open(self) -> bool:
        return bool(self._call("GET", f"{PAPER_URL}/v2/clock")["is_open"])

    def held_qty(self, symbol: str) -> float:
        """Shares held in ``symbol``: negative for a short, zero when there is no position."""
        data = self._call("GET", f"{PAPER_URL}/v2/positions/{symbol}", missing_ok=True)
        return float(data["qty"]) if data else 0.0

    def open_orders(self, symbol: str) -> int:
        orders = self._call("GET", f"{PAPER_URL}/v2/orders",
                            params={"status": "open", "symbols": symbol, "limit": 100})
        return len(orders)

    def last_price(self, symbol: str) -> float:
        data = self._call("GET", f"{DATA_URL}/v2/stocks/{symbol}/trades/latest")
        return float(data["trade"]["p"])

    def submit_bracket(self, order: BracketOrder, client_order_id: str) -> dict:
        """Send a market buy with its stop-loss and take-profit attached.

        Good till cancelled: the exits take the entry's time in force, and a day
        order's exits would lapse at the close and leave the position open with
        neither. ``client_order_id`` makes the request safe to repeat: the
        broker refuses a second order under an id it has seen.
        """
        return self._call("POST", f"{PAPER_URL}/v2/orders", json={
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "bracket",
            "take_profit": {"limit_price": f"{order.take_profit_price:.2f}"},
            "stop_loss": {"stop_price": f"{order.stop_price:.2f}"},
            "client_order_id": client_order_id,
        })
