# src/cryptotrader/integrations/mt5/__init__.py
"""MetaTrader 5 bridge.

A MQL5 Expert Advisor (``expert/CryptoTraderBridge.mq5``) runs inside an MT5
terminal and talks to this service over HTTP:

* it POSTs its broker bars + current position to ``/api/mt5/decide`` and
* receives a trading decision (open/close/hold + protective levels), which it
  executes on the broker itself.

That keeps the ML/decision brain unchanged on Linux/Coolify while MT5 (Windows)
handles only data capture and order execution. See ``docs/mt5.md``.
"""

from cryptotrader.integrations.mt5.bridge import decide, map_symbol

__all__ = ["decide", "map_symbol"]
