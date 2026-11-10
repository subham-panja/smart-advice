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

### B. Order Placement Implementation
```python
from dhanhq import dhanhq
from brokers.base_broker import BaseBroker
import config

class DhanBrokerClient(BaseBroker):
    def __init__(self):
        self.client_id = config.DHAN_CONFIG.get("client_id")
        self.access_token = config.DHAN_CONFIG.get("access_token")
        self.dhan = None
        self._security_id_map = {}

    def connect(self) -> bool:
        try:
            self.dhan = dhanhq(self.client_id, self.access_token)
            fund_limits = self.dhan.get_fund_limits()
            return fund_limits.get("status") == "success"
        except Exception as e:
            logger.error(f"Dhan connection failed: {e}")
            return False

    def place_order(self, symbol, transaction_type, quantity, order_type="MARKET", price=0.0, product_type="CNC", tag="SWING"):
        sec_id = self._get_security_id(symbol)
        dhan_txn = self.dhan.BUY if transaction_type.upper() == "BUY" else self.dhan.SELL
        dhan_ord = self.dhan.MARKET if order_type.upper() == "MARKET" else self.dhan.LIMIT
        dhan_prod = self.dhan.CNC if product_type == "CNC" else self.dhan.INTRA

        res = self.dhan.place_order(
            security_id=sec_id,
            exchange_segment=self.dhan.NSE_EQ,
            transaction_type=dhan_txn,
            quantity=quantity,
            order_type=dhan_ord,
            product_type=dhan_prod,
            price=price,
            correlation_id=tag
        )
        return res
```

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
