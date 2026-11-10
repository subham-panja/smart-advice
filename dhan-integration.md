# DhanHQ API Integration Plan (Broker Adaptable Pattern)

This document outlines the architectural plan to integrate **DhanHQ API (v2)** into our trading orchestrator and execution pipeline while maintaining a **Broker-Agnostic Adapter Pattern** alongside our existing **5paisa** and **Paper Trading** engines.

---

## 1. Architectural Strategy: The Broker Adapter Pattern

Currently, our system directly references `fivepaisa_client.py` and paper trading fallbacks. To allow hot-swapping or dual routing between **5paisa**, **Dhan**, and **Paper Trading** via configuration without rewriting orchestrator logic, we will implement an abstract base class `BrokerClient` interface.

```
                  ┌──────────────────────────────┐
                  │   Trading Cycle Orchestrator │
                  │     (main_orchestrator.py)   │
                  └──────────────┬───────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │ Broker Factory / Router│
                     │  (broker_factory.py)  │
                     └───────────┬───────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     │                           │                           │
┌────▼─────────────────┐   ┌─────▼────────────────┐   ┌──────▼─────────────────┐
│ PaperTradingClient   │   │ DhanBrokerClient     │   │ FivePaisaBrokerClient  │
│ (In-memory/MongoDB)  │   │ (dhanhq SDK v2)      │   │ (py5paisa SDK)         │
└──────────────────────┘   └──────────────────────┘   └────────────────────────┘
```

---

## 2. DhanHQ API Specifications

* **Official Python SDK**: `dhanhq` (`pip install dhanhq`)
* **REST Base URL**: `https://api.dhan.co/v2`
* **WebSocket Feeds**: `wss://api-feed.dhan.co`
* **Authentication**: 
  * `DHAN_CLIENT_ID`: 10-digit Dhan Client ID
  * `DHAN_ACCESS_TOKEN`: JWT Auth token from Dhan Web Developer Portal (valid for 30 days)

### Required Environment Variables (`.env`)
```bash
# Broker Provider Switch ('paper' | '5paisa' | 'dhan')
ACTIVE_BROKER=dhan

# DhanHQ Credentials
DHAN_CLIENT_ID=1000000000
DHAN_ACCESS_TOKEN=your_dhan_access_token_jwt
```

---

## 3. Abstract Broker Interface (`backend/brokers/base_broker.py`)

All broker implementations will adhere to this contract:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BaseBroker(ABC):
    @abstractmethod
    def connect(self) -> bool:
        """Authenticate and verify connectivity with broker."""
        pass

    @abstractmethod
    def get_funds(self) -> Dict[str, float]:
        """Return available cash, margin used, and total collateral balance."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """Fetch current open net positions and unrealized PnL."""
        pass

    @abstractmethod
    def get_holdings(self) -> List[Dict]:
        """Fetch long-term portfolio holdings from demat."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        transaction_type: str,  # 'BUY' | 'SELL'
        quantity: int,
        order_type: str = "MARKET",  # 'MARKET' | 'LIMIT'
        price: float = 0.0,
        trigger_price: float = 0.0,
        product_type: str = "CNC",  # 'CNC' (delivery) | 'INTRADAY'
        tag: str = "SMART_ADVICE"
    ) -> Dict:
        """Place live or simulated trade order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open pending order."""
        pass
```

---

## 4. Dhan Implementation (`backend/brokers/dhan_client.py`)

### A. Scrip & Security ID Resolution
Dhan requires a numeric `security_id` (e.g. `1333` for `RELIANCE`) rather than standard ticker names.
* We download Dhan's daily scrip master CSV (`https://images.dhan.co/api-data/api-scrip-master.csv`) and index `SEM_TRADING_SYMBOL` -> `SEM_SMST_SECURITY_ID`.
* Cache the master mapping locally under `backend/data/dhan_scrip_master.json`.

### B. Order Placement & Forever (GTT OCO) Orders

Dhan supports native **Forever Orders** (Good-Till-Triggered) with **OCO (One-Cancels-the-Other)** brackets. This is the optimal architecture for swing trading:

1. **Entry Leg (Single Trigger GTT)**:
   * Waits indefinitely until price reaches our technical entry breakout pivot (`trigger_price = buy_price`).
   * No capital is blocked until the breakout actually triggers.
2. **Post-Fill Exit Leg (OCO Forever Order)**:
   * Automatically sets **Target Limit (Leg 1)** and **Stop Loss Trigger (Leg 2)** at exchange level.
   * If Target 1 triggers, Stop Loss is canceled automatically (and vice-versa).
   * Runs natively on Dhan's servers, eliminating server uptime dependencies for stop protection.

```python
def place_forever_bracket(self, symbol, quantity, buy_price, stop_loss, target):
    """Place a native Dhan GTT OCO order for target and stop-loss management."""
    sec_id = self._get_security_id(symbol)
    return self.dhan.place_forever(
        security_id=sec_id,
        exchange_segment=self.dhan.NSE,
        transaction_type=dhan.SELL,
        order_type=dhan.LIMIT,
        product_type=dhan.CNC,
        quantity=quantity,
        price=target,
        trigger_price=target * 0.998,
        order_flag="OCO",
        price1=stop_loss * 0.995,
        trigger_price1=stop_loss,
        quantity1=quantity,
    )
```

---

## 4.1 Real-Time Pricing vs. Previous-Day EOD Gap (Solution)

### The Core Problem:
Currently, the screening cycle uses EOD daily bars. If a stock opens gap-up +4% or rallies early morning, a static limit order anchored to yesterday's close fails or misses execution entirely.

### The 3-Step Real-Time Solution:
1. **Pre-Execution Real-Time LTP Check (`dhan.get_ltp`)**:
   * Before sending an order to Dhan or paper engine, call `dhan.get_ltp([(exchange_segment, sec_id)])`.
   * Compare Live LTP against the Recommendation `buy_price`:
     * If `live_ltp <= buy_price * 1.015` (within 1.5% slippage tolerance): Execute order at `live_ltp`.
     * If `live_ltp > buy_price * 1.025` (stock already ran up 2.5%+): **Abort entry**, mark status as `CHASE_AVOIDED` (never chase extended stocks).
     * If `live_ltp < buy_price` (stock consolidating below breakout pivot): Deploy a **Dhan Forever Entry Order** (`trigger_price = buy_price`), letting Dhan trigger the buy the exact minute price breaks out.
2. **Live Account Balance Sync (`dhan.get_fund_limits`)**:
   * Pulls real-time `availMargin` directly from Dhan instead of hardcoded config.
   * Calculates position sizing dynamically: `Quantity = min(Risk_Amount / (Entry - SL), Avail_Margin * 0.125 / Entry)`.
3. **Live WebSocket Ticker (`marketfeed.DhanFeed`)**:
   * Subscribes open positions to Dhan's low-latency WebSocket feed to maintain real-time PnL on frontend without polling delays.

---

## 5. Configuration Switch in `backend/config.py`

```python
# Broker selector: 'paper' | '5paisa' | 'dhan'
ACTIVE_BROKER = os.getenv("ACTIVE_BROKER", "paper")

DHAN_CONFIG = {
    "client_id": os.getenv("DHAN_CLIENT_ID"),
    "access_token": os.getenv("DHAN_ACCESS_TOKEN"),
    "scrip_master_url": "https://images.dhan.co/api-data/api-scrip-master.csv",
}
```

---

## 6. Execution Milestones & Rollout Plan

| Phase | Deliverable | Details |
| :--- | :--- | :--- |
| **Phase 1** | Abstract Broker Interface | Create `BaseBroker` contract and wrap current `FivePaisaClient` & `PaperTradingClient`. |
| **Phase 2** | Dhan Scrip Master Sync | Build automatic daily downloader for `api-scrip-master.csv` to map NSE symbols -> Dhan `security_id`. |
| **Phase 3** | DhanBrokerClient Implementation | Implement `dhanhq` order routing, position tracking, and fund limit calls. |
| **Phase 4** | Orchestrator Integration | Refactor `main_orchestrator.py` and `run_analysis.py` to route orders through `get_active_broker()`. |
| **Phase 5** | Live Shadow Testing & Verification | Test order execution in Paper mode, verify Dhan balance display in frontend Settings and Terminal. |
EOF
