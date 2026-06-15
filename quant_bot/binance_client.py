"""Minimal Binance REST API client for live/testnet trading.

Uses ONLY the standard library + requests (already a dependency).  No need for
the python-binance third-party package — that keeps the install footprint small
and the code easy to audit.

Sign up for free testnet keys at: https://testnet.binance.vision/
"""
from __future__ import annotations  # newer type-hint syntax on older Python

import hashlib  # standard library — for SHA256 hashing used in HMAC signing
import hmac     # standard library — implements HMAC (Hash-based Message Authentication Code) signing required by Binance
import logging  # for structured timestamped log output
import time     # to get current Unix time in milliseconds for the request timestamp
from typing import Any  # type hint for "any JSON-shaped value"

import requests  # third-party HTTP library — used for the REST calls
from requests.adapters import HTTPAdapter  # to attach retry behaviour to the session
from urllib3.util.retry import Retry  # automatic retry policy for transient errors

logger = logging.getLogger(__name__)  # logger named "quant_bot.binance_client"


class BinanceClient:  # thin wrapper over the Binance Spot REST API — handles signing and HTTP for us
    """Lightweight Binance REST client supporting testnet and production.

    Usage:
        client = BinanceClient(api_key, api_secret, testnet=True)
        client.get_account()                           # check balances
        client.place_market_order("BTCUSDT", "BUY", quantity=0.001)
    """

    def __init__(self, api_key: str, api_secret: str, *, testnet: bool = True) -> None:  # constructor — store credentials and choose endpoint
        # Lazy import here so the module loads even if these config values are missing during dev
        from quant_bot.config import BINANCE_LIVE_REST_URL, BINANCE_TESTNET_REST_URL  # import inside function to avoid circular imports at top-level

        self.api_key = api_key  # Binance API key — public identifier of your account
        self.api_secret = api_secret.encode("utf-8")  # API secret as bytes — HMAC requires bytes; .encode("utf-8"): convert string to UTF-8 bytes
        self.base_url = BINANCE_TESTNET_REST_URL if testnet else BINANCE_LIVE_REST_URL  # pick testnet or production URL based on the flag
        self.testnet = testnet  # remember which mode for logging

        # Build a requests Session with auto-retry on transient errors (same pattern as data.py)
        self.session = requests.Session()  # Session: persistent HTTP connection — reuses settings across requests
        retry = Retry(  # retry configuration
            total=3,                       # try the request up to 3 extra times
            backoff_factor=0.5,            # wait 0.5s, 1s, 2s between retries (exponential back-off)
            status_forcelist=[429, 500, 502, 503, 504],  # retry on rate-limit and server errors
            allowed_methods=["GET"],       # only retry GET requests — never retry POST (would risk duplicate orders!)
            raise_on_status=False,         # don't raise inside retry — handle manually
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))  # attach retry to all https requests
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})  # X-MBX-APIKEY: Binance's required auth header on every authenticated endpoint

        env_label = "TESTNET" if testnet else "LIVE (REAL MONEY)"  # human-readable label for the log
        logger.info("BinanceClient initialised → %s (%s)", env_label, self.base_url)  # confirm which endpoint we're using

    # ── Public (unsigned) endpoints ───────────────────────────────────────────

    def get_server_time(self) -> int:  # GET /api/v3/time — returns Binance's server time in ms; used to sync our clock with theirs
        response = self.session.get(f"{self.base_url}/api/v3/time", timeout=10)  # send GET; timeout: max 10s for a reply
        response.raise_for_status()  # raise on HTTP error
        return int(response.json()["serverTime"])  # response.json(): parse JSON body; ["serverTime"]: extract the integer timestamp

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> list[list[Any]]:  # GET candlestick data — same endpoint as in data.py but exposed here for the live trader
        params = {"symbol": symbol, "interval": interval, "limit": limit}  # query params
        response = self.session.get(f"{self.base_url}/api/v3/klines", params=params, timeout=15)  # send request
        response.raise_for_status()
        return response.json()  # returns a list of candle lists matching the data.py format

    # ── Private (signed) endpoints ────────────────────────────────────────────

    def _sign(self, params: dict[str, Any]) -> dict[str, Any]:  # internal helper — adds timestamp and HMAC signature to a params dict
        params["timestamp"] = int(time.time() * 1000)  # required by Binance — current Unix time in ms
        params["recvWindow"] = 5000  # max ms our request can be delayed before Binance rejects it (5s = generous default)
        # Build the query string the same way requests will: key1=value1&key2=value2…
        query = "&".join(f"{k}={v}" for k, v in params.items())  # join params as URL-encoded-style key=value pairs
        # HMAC-SHA256 sign the query string with the API secret — Binance verifies this proves we own the secret
        signature = hmac.new(  # hmac.new(): create an HMAC object
            self.api_secret,             # key — our secret (already bytes)
            query.encode("utf-8"),       # message — the query string as bytes
            hashlib.sha256,              # hash function — SHA256
        ).hexdigest()  # .hexdigest(): get the signature as a hex string Binance can verify
        params["signature"] = signature  # add signature as the final param
        return params  # return the signed params ready to be sent

    def get_account(self) -> dict[str, Any]:  # GET /api/v3/account — returns balances and account status
        params = self._sign({})  # sign an empty param dict (account endpoint needs no params besides timestamp/signature)
        response = self.session.get(f"{self.base_url}/api/v3/account", params=params, timeout=10)  # send GET to the signed endpoint
        response.raise_for_status()
        return response.json()  # returns dict with "balances" list, "canTrade" bool, etc.

    def get_balance(self, asset: str) -> float:  # convenience — extract just one asset's free balance
        account = self.get_account()  # fetch the full account snapshot
        for bal in account.get("balances", []):  # loop balances list; .get("balances", []): empty list if key missing
            if bal["asset"] == asset:  # find the row for the asset we want (e.g. "USDT" or "BTC")
                return float(bal["free"])  # "free": amount available for trading (not locked in open orders)
        return 0.0  # asset not held — balance is 0

    def place_market_order(  # POST /api/v3/order — place a MARKET order (fills immediately at best available price)
        self,
        symbol: str,           # e.g. "BTCUSDT"
        side: str,             # "BUY" or "SELL"
        quantity: float | None = None,    # quantity in BASE asset (BTC) — use for SELL or "I want exactly this much BTC" BUY
        quote_quantity: float | None = None,  # quantity in QUOTE asset (USDT) — use for "I want to spend exactly this many USD" BUY
    ) -> dict[str, Any]:
        if (quantity is None) == (quote_quantity is None):  # XOR check — exactly one of the two must be provided
            raise ValueError("Pass either quantity OR quote_quantity, not both/neither.")

        params: dict[str, Any] = {  # base params for the order request
            "symbol": symbol,
            "side": side,
            "type": "MARKET",  # MARKET: fills immediately at the current best price (vs LIMIT which waits for a target price)
        }
        if quantity is not None:  # caller specified BTC amount
            params["quantity"] = f"{quantity:.6f}"  # 6 decimal places — Binance requires properly formatted numbers
        else:  # caller specified USDT amount
            params["quoteOrderQty"] = f"{quote_quantity:.2f}"  # quoteOrderQty: spend exactly this much USDT (Binance computes BTC amount automatically)

        signed = self._sign(params)  # add timestamp + HMAC signature
        logger.info("ORDER → %s %s %s | params=%s", "TESTNET" if self.testnet else "LIVE", side, symbol, signed)  # log the order before sending (signature is benign to log on testnet)
        response = self.session.post(f"{self.base_url}/api/v3/order", params=signed, timeout=15)  # POST: actually place the order
        if response.status_code != 200:  # non-200 = order rejected — log the error message for debugging
            logger.error("Order failed (%d): %s", response.status_code, response.text)
            response.raise_for_status()  # raise so the caller knows the order did not go through
        result = response.json()  # parse response: includes order ID, fills, total executed quantity
        logger.info("Order filled: id=%s status=%s qty=%s", result.get("orderId"), result.get("status"), result.get("executedQty"))
        return result  # return the full order response — caller can inspect fills
