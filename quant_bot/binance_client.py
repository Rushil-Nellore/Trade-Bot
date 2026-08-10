"""Minimal Binance REST API client for live/testnet trading.

Uses ONLY the standard library + requests (already a dependency).  No need for
the python-binance third-party package — that keeps the install footprint small
and the code easy to audit.

Sign up for free testnet keys at: https://testnet.binance.vision/
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class BinanceClient:
    """Lightweight Binance REST client supporting testnet and production.

    Usage:
        client = BinanceClient(api_key, api_secret, testnet=True)
        client.get_account()                           # check balances
        client.place_market_order("BTCUSDT", "BUY", quantity=0.001)
    """

    def __init__(self, api_key: str, api_secret: str, *, testnet: bool = True) -> None:
        from quant_bot.config import BINANCE_LIVE_REST_URL, BINANCE_TESTNET_REST_URL

        self.api_key = api_key
        self.api_secret = api_secret.encode("utf-8")
        self.base_url = BINANCE_TESTNET_REST_URL if testnet else BINANCE_LIVE_REST_URL
        self.testnet = testnet

        self.session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

        env_label = "TESTNET" if testnet else "LIVE (REAL MONEY)"
        logger.info("BinanceClient initialised → %s (%s)", env_label, self.base_url)

    def get_server_time(self) -> int:
        response = self.session.get(f"{self.base_url}/api/v3/time", timeout=10)
        response.raise_for_status()
        return int(response.json()["serverTime"])

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> list[list[Any]]:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = self.session.get(f"{self.base_url}/api/v3/klines", params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = "&".join(f"{k}={v}" for k, v in params.items())
        signature = hmac.new(
            self.api_secret,
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def get_account(self) -> dict[str, Any]:
        params = self._sign({})
        response = self.session.get(f"{self.base_url}/api/v3/account", params=params, timeout=10)
        response.raise_for_status()
        return response.json()

    def get_balance(self, asset: str) -> float:
        account = self.get_account()
        for bal in account.get("balances", []):
            if bal["asset"] == asset:
                return float(bal["free"])
        return 0.0

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float | None = None,
        quote_quantity: float | None = None,
    ) -> dict[str, Any]:
        if (quantity is None) == (quote_quantity is None):
            raise ValueError("Pass either quantity OR quote_quantity, not both/neither.")

        params: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
        }
        if quantity is not None:
            params["quantity"] = f"{quantity:.6f}"
        else:
            params["quoteOrderQty"] = f"{quote_quantity:.2f}"

        signed = self._sign(params)
        logger.info("ORDER → %s %s %s | params=%s", "TESTNET" if self.testnet else "LIVE", side, symbol, signed)
        response = self.session.post(f"{self.base_url}/api/v3/order", params=signed, timeout=15)
        if response.status_code != 200:
            logger.error("Order failed (%d): %s", response.status_code, response.text)
            response.raise_for_status()
        result = response.json()
        logger.info("Order filled: id=%s status=%s qty=%s", result.get("orderId"), result.get("status"), result.get("executedQty"))
        return result
