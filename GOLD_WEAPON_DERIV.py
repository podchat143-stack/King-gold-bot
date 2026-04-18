"""
================================================================================
TFX GOLD SNIPER - SULTAN EDITION v2.0 (DERIV API)
================================================================================
Original: MT5 based
Rewritten: Deriv WebSocket API — runs on Railway.app (Linux)
Same logic: D-RSI + SMC + Kelly + Telegram + Session Filter

SETUP:
Set these environment variables in Railway.app:
  DERIV_API_TOKEN   = your Deriv API token
  DERIV_ACCOUNT_ID  = your CR number (e.g. CR1234567)
  TELEGRAM_BOT_TOKEN = your Telegram bot token
  TELEGRAM_CHAT_ID   = your Telegram chat ID
  DERIV_IS_DEMO      = true (for demo) or false (for live)
================================================================================
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import time
import logging
import requests
import os
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

# ================= ⚙️ CONFIGURATION =================
DERIV_API_TOKEN   = os.environ.get("DERIV_API_TOKEN", "YOUR_API_TOKEN")
DERIV_ACCOUNT_ID  = os.environ.get("DERIV_ACCOUNT_ID", "YOUR_ACCOUNT_ID")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
IS_DEMO            = os.environ.get("DERIV_IS_DEMO", "true").lower() == "true"

SYMBOL        = "frxXAUUSD"       # Deriv symbol for Gold
GRANULARITY   = 300               # M5 = 300 seconds
HTF_GRAN      = 3600              # H1 = 3600 seconds
MAGIC_COMMENT = "SultanGold"

# --- Core D-RSI Parameters ---
DRSI_WINDOW        = 28
DRSI_DEGREE        = 2
DRSI_RSI_LENGTH    = 14
DRSI_SIGNAL_LENGTH = 2

# --- Entry/Exit Conditions ---
BUY_CONDITION  = "Signal Line Crossing"
SELL_CONDITION = "Signal Line Crossing"
EXIT_CONDITION = "Signal Line Crossing"

# --- RMSE Filter ---
USE_RMSE_FILTER  = True
RMSE_THRESHOLD   = 0.12

# --- SMC Confluence ---
USE_SMC_CONFLUENCE = True
SMC_FVG_LOOKBACK   = 50
SMC_OB_LOOKBACK    = 20

# --- Multi-Timeframe Filter ---
USE_HTF_TREND_FILTER = True
HTF_TREND_MA_PERIOD  = 50

# --- Session Filter ---
USE_SESSION_FILTER = True
LONDON_START = 8
LONDON_END   = 17
NY_START     = 13
NY_END       = 22

# --- News Filter ---
USE_NEWS_FILTER    = True
NEWS_QUIET_MINUTES = 30

# --- Risk Management ---
RISK_PERCENT         = 1.0
USE_KELLY_SIZING     = True
KELLY_FRACTION       = 0.25
MAX_DAILY_LOSS_PCT   = 6.0
MAX_DRAWDOWN_PCT     = 15.0
MAX_SL_PIPS          = 60
MIN_SL_PIPS          = 12
DESIRED_RR           = 2.0
PIP_VALUE            = 0.01      # For XAUUSD

# --- Trailing Stop ---
USE_TRAILING_STOP       = True
TRAILING_TRIGGER_PIPS   = 18
TRAILING_STEP_PIPS      = 6
MIN_TRAIL_DISTANCE_PIPS = 10

# --- Candle buffer size ---
CANDLE_BUFFER = 200

# ===============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"

# ================= UTILITY CLASSES =================

class MarketSession(Enum):
    ASIA    = "Asia"
    LONDON  = "London"
    NY      = "New York"
    OVERLAP = "London/NY Overlap"
    CLOSED  = "Closed"

class TradeDirection(Enum):
    BUY  = "BUY"
    SELL = "SELL"

# ================= TELEGRAM =================

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token    = token
        self.chat_id  = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled  = bool(token and chat_id and token != "YOUR_BOT_TOKEN")
        if not self.enabled:
            print("⚠️ Telegram not configured.")

    def send_message(self, text: str, parse_mode: str = "HTML"):
        if not self.enabled:
            return
        try:
            url     = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logging.error(f"Telegram error: {e}")

# ================= NEWS FILTER =================

class NewsFilter:
    @staticmethod
    def is_news_time(current_time: datetime, quiet_minutes: int = 30) -> bool:
        # TODO: plug in real ForexFactory API
        return False

# ================= MATRIX MATH =================

class MatrixOps:
    @staticmethod
    def vandermonde(window: int, degree: int) -> np.ndarray:
        return np.vander(np.arange(window), degree + 1, increasing=True)

    @staticmethod
    def pseudo_inverse(A: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(A)

    @staticmethod
    def polynomial_derivative(y: np.ndarray, window: int, degree: int) -> Tuple[float, float]:
        J        = MatrixOps.vandermonde(window, degree + 1)
        J_pinv   = MatrixOps.pseudo_inverse(J)
        a_coef   = J_pinv @ y
        deriv    = 0.0
        for i in range(1, degree + 1):
            deriv += i * a_coef[i] * ((window - 1) ** (i - 1))
        y_hat = J @ a_coef
        mse   = np.mean((y - y_hat) ** 2)
        rmse  = np.sqrt(mse)
        nrmse = rmse / np.mean(y) if np.mean(y) != 0 else 0.0
        return deriv, nrmse

# ================= SMC CONFLUENCE =================

class SMCDetector:
    def detect_fvg(self, df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        df = df.copy()
        df['bull_fvg'] = (df['low'].shift(2) > df['high']) & (df['close'].shift(1) > df['open'].shift(1))
        df['bear_fvg'] = (df['high'].shift(2) < df['low']) & (df['close'].shift(1) < df['open'].shift(1))
        return df

    def detect_order_blocks(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        df = df.copy()
        df['bull_ob'] = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['high'].shift(1))
        df['bear_ob'] = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['low'].shift(1))
        return df

    def is_bullish_confluence(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < 5:
            return False
        recent = df.iloc[max(0, idx - 10):idx + 1]
        return recent['bull_fvg'].any() or recent['bull_ob'].any()

    def is_bearish_confluence(self, df: pd.DataFrame, idx: int) -> bool:
        if idx < 5:
            return False
        recent = df.iloc[max(0, idx - 10):idx + 1]
        return recent['bear_fvg'].any() or recent['bear_ob'].any()

# ================= D-RSI ENGINE =================

class DRSIEngine:
    def __init__(self, window, degree, rsi_len, signal_len):
        self.window        = window
        self.degree        = degree
        self.rsi_len       = rsi_len
        self.signal_len    = signal_len
        self.prev_drsi     = 0.0
        self.prev_drsi_2   = 0.0
        self.prev_signal   = 0.0
        self.drsi_history  = []

    def calculate_rsi(self, prices: np.ndarray) -> np.ndarray:
        deltas = np.diff(prices)
        seed   = deltas[:self.rsi_len + 1]
        up     = seed[seed >= 0].sum() / self.rsi_len
        down   = -seed[seed < 0].sum() / self.rsi_len
        rs     = up / down if down != 0 else 0
        rsi    = np.zeros_like(prices)
        rsi[:self.rsi_len] = 100. - 100. / (1. + rs) if down != 0 else 100
        for i in range(self.rsi_len, len(prices)):
            delta   = deltas[i - 1]
            upval   = delta if delta > 0 else 0
            downval = -delta if delta < 0 else 0
            up      = (up * (self.rsi_len - 1) + upval) / self.rsi_len
            down    = (down * (self.rsi_len - 1) + downval) / self.rsi_len
            rs      = up / down if down != 0 else 0
            rsi[i]  = 100. - 100. / (1. + rs) if down != 0 else 100
        return rsi

    def process(self, close_prices: np.ndarray) -> Optional[Dict[str, Any]]:
        if len(close_prices) < self.rsi_len + self.window:
            return None
        rsi_vals   = self.calculate_rsi(close_prices)
        rsi_window = rsi_vals[-self.window:]
        drsi, nrmse = MatrixOps.polynomial_derivative(rsi_window, self.window, self.degree)
        self.drsi_history.append(drsi)
        if len(self.drsi_history) > self.signal_len + 2:
            self.drsi_history.pop(0)
        alpha       = 2.0 / (self.signal_len + 1) if self.signal_len > 0 else 1.0
        signal_line = self.prev_signal * (1 - alpha) + drsi * alpha if self.prev_signal != 0 else drsi

        dir_up      = (drsi > self.prev_drsi) and (self.prev_drsi < self.prev_drsi_2) and self.prev_drsi < 0
        dir_dw      = (drsi < self.prev_drsi) and (self.prev_drsi > self.prev_drsi_2) and self.prev_drsi > 0
        cross_up    = (self.prev_drsi <= 0 and drsi > 0)
        cross_dw    = (self.prev_drsi >= 0 and drsi < 0)
        cross_sig_up = (self.prev_drsi <= self.prev_signal and drsi > signal_line)
        cross_sig_dw = (self.prev_drsi >= self.prev_signal and drsi < signal_line)

        self.prev_drsi_2 = self.prev_drsi
        self.prev_drsi   = drsi
        self.prev_signal = signal_line

        return {
            'drsi':         drsi,
            'signal_line':  signal_line,
            'nrmse':        nrmse,
            'rsi':          rsi_vals[-1],
            'dir_up':       dir_up,
            'dir_dw':       dir_dw,
            'cross_up':     cross_up,
            'cross_dw':     cross_dw,
            'cross_sig_up': cross_sig_up,
            'cross_sig_dw': cross_sig_dw,
            'filter_pass':  nrmse < RMSE_THRESHOLD if USE_RMSE_FILTER else True
        }

# ================= DERIV API CLIENT =================

class DerivClient:
    """Handles all Deriv WebSocket communication"""

    def __init__(self, token: str):
        self.token   = token
        self.ws      = None
        self.req_id  = 1
        self._pending: Dict[int, asyncio.Future] = {}

    async def connect(self):
        self.ws = await websockets.connect(DERIV_WS_URL)
        asyncio.ensure_future(self._listener())
        # Authorize
        resp = await self.send({"authorize": self.token})
        if resp.get("error"):
            raise Exception(f"Auth failed: {resp['error']['message']}")
        logging.info(f"✅ Deriv authorized: {resp['authorize']['loginid']}")
        return resp

    async def _listener(self):
        async for message in self.ws:
            data = json.loads(message)
            req_id = data.get("req_id")
            if req_id and req_id in self._pending:
                future = self._pending.pop(req_id)
                if not future.done():
                    future.set_result(data)

    async def send(self, payload: dict) -> dict:
        req_id = self.req_id
        self.req_id += 1
        payload["req_id"] = req_id
        future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = future
        await self.ws.send(json.dumps(payload))
        return await asyncio.wait_for(future, timeout=30)

    async def get_candles(self, symbol: str, granularity: int, count: int) -> List[dict]:
        resp = await self.send({
            "ticks_history": symbol,
            "style":         "candles",
            "granularity":   granularity,
            "count":         count,
            "end":           "latest"
        })
        if resp.get("error"):
            logging.error(f"Candle fetch error: {resp['error']['message']}")
            return []
        return resp.get("candles", [])

    async def get_tick(self, symbol: str) -> Optional[dict]:
        resp = await self.send({"ticks": symbol, "subscribe": 0})
        if resp.get("error"):
            return None
        return resp.get("tick")

    async def get_account_info(self) -> Optional[dict]:
        resp = await self.send({"balance": 1, "subscribe": 0})
        if resp.get("error"):
            return None
        return resp.get("balance")

    async def get_open_contracts(self) -> List[dict]:
        resp = await self.send({"portfolio": 1})
        if resp.get("error"):
            return []
        return resp.get("portfolio", {}).get("contracts", [])

    async def buy_contract(self, direction: str, amount: float, symbol: str,
                           duration: int, duration_unit: str,
                           barrier: Optional[str] = None) -> Optional[dict]:
        """
        Place a trade using Deriv Multipliers (CFD-like for MT5 accounts)
        For CFD/MT5 accounts on Deriv, we use the trading_servers endpoint.
        This uses vanilla buy for simplicity.
        """
        contract_type = "CALL" if direction == "BUY" else "PUT"
        payload = {
            "buy": 1,
            "price": amount,
            "parameters": {
                "amount":          amount,
                "basis":           "stake",
                "contract_type":   contract_type,
                "currency":        "USD",
                "duration":        duration,
                "duration_unit":   duration_unit,
                "symbol":          symbol,
            }
        }
        resp = await self.send(payload)
        if resp.get("error"):
            logging.error(f"Buy failed: {resp['error']['message']}")
            return None
        return resp.get("buy")

    async def sell_contract(self, contract_id: int, price: float = 0) -> Optional[dict]:
        resp = await self.send({"sell": contract_id, "price": price})
        if resp.get("error"):
            logging.error(f"Sell failed: {resp['error']['message']}")
            return None
        return resp.get("sell")

    async def get_profit_table(self) -> List[dict]:
        resp = await self.send({"profit_table": 1, "description": 1, "limit": 20})
        if resp.get("error"):
            return []
        return resp.get("profit_table", {}).get("transactions", [])

# ================= MAIN SULTAN BOT =================

class SultanGoldBot:
    def __init__(self):
        self.client              = DerivClient(DERIV_API_TOKEN)
        self.telegram            = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.drsi                = DRSIEngine(DRSI_WINDOW, DRSI_DEGREE, DRSI_RSI_LENGTH, DRSI_SIGNAL_LENGTH)
        self.smc                 = SMCDetector()
        self.news_filter         = NewsFilter()
        self.daily_start_balance = 0.0
        self.peak_equity         = 0.0
        self.is_running          = True
        self.last_candle_time    = None
        self.open_contracts      = []
        self.trade_count         = 0

    def get_session(self, dt: datetime = None) -> MarketSession:
        if dt is None:
            dt = datetime.utcnow()
        hour = dt.hour
        if LONDON_START <= hour < LONDON_END:
            if NY_START <= hour < NY_END:
                return MarketSession.OVERLAP
            return MarketSession.LONDON
        elif NY_START <= hour < NY_END:
            return MarketSession.NY
        return MarketSession.ASIA

    async def is_trading_allowed(self) -> bool:
        if USE_SESSION_FILTER:
            sess = self.get_session()
            if sess == MarketSession.ASIA:
                logging.info("🔕 Asia session — no trading")
                return False
        if USE_NEWS_FILTER:
            if self.news_filter.is_news_time(datetime.utcnow(), NEWS_QUIET_MINUTES):
                logging.info("🔕 News filter active")
                return False
        acc = await self.client.get_account_info()
        if acc:
            balance = acc.get("balance", 0)
            if balance <= 0:
                return False
            equity = balance  # Deriv balance = equity for options
            if self.daily_start_balance > 0:
                daily_pnl_pct = (equity - self.daily_start_balance) / self.daily_start_balance * 100
                if daily_pnl_pct <= -MAX_DAILY_LOSS_PCT:
                    logging.warning("⛔ Daily loss limit hit")
                    self.telegram.send_message("⛔ <b>Daily loss limit hit. Trading paused.</b>")
                    return False
            if equity > self.peak_equity:
                self.peak_equity = equity
            if self.peak_equity > 0:
                dd_pct = (self.peak_equity - equity) / self.peak_equity * 100
                if dd_pct >= MAX_DRAWDOWN_PCT:
                    logging.warning("⛔ Max drawdown hit")
                    self.telegram.send_message("⛔ <b>Max drawdown hit. Trading paused.</b>")
                    return False
        return True

    async def get_htf_trend(self) -> Optional[str]:
        candles = await self.client.get_candles(SYMBOL, HTF_GRAN, HTF_TREND_MA_PERIOD + 50)
        if len(candles) < HTF_TREND_MA_PERIOD:
            return None
        closes = np.array([c['close'] for c in candles])
        ema    = pd.Series(closes).ewm(span=HTF_TREND_MA_PERIOD, adjust=False).mean().iloc[-1]
        price  = closes[-1]
        if price > ema * 1.002:
            return 'bull'
        elif price < ema * 0.998:
            return 'bear'
        return None

    def calculate_stake(self, sl_pips: float, win_prob: float = 0.45) -> float:
        """Kelly-based stake sizing"""
        if USE_KELLY_SIZING:
            R         = DESIRED_RR
            p         = win_prob
            kelly_pct = p - (1 - p) / R
            kelly_pct = max(0, min(kelly_pct, 0.05))
            risk_pct  = kelly_pct * KELLY_FRACTION
        else:
            risk_pct = RISK_PERCENT / 100.0
        balance      = self.daily_start_balance if self.daily_start_balance > 0 else 100.0
        stake        = balance * risk_pct
        stake        = max(1.0, round(stake, 2))
        return stake

    def get_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period=14) -> float:
        if len(close) < period + 1:
            return 0.0
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr  = np.maximum(np.maximum(tr1, tr2), tr3)
        return np.mean(tr[-period:])

    async def analyze_and_trade(self):
        # Fetch M5 candles
        candles = await self.client.get_candles(SYMBOL, GRANULARITY, DRSI_WINDOW + DRSI_RSI_LENGTH + 50)
        if len(candles) < DRSI_WINDOW + DRSI_RSI_LENGTH:
            logging.warning("Not enough candles")
            return

        df = pd.DataFrame(candles)
        df.rename(columns={'epoch': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        close = df['close'].values.astype(float)
        high  = df['high'].values.astype(float)
        low   = df['low'].values.astype(float)

        # D-RSI signal
        drsi_res = self.drsi.process(close)
        if drsi_res is None:
            logging.warning("D-RSI returned None")
            return

        # Entry signals
        if BUY_CONDITION == "Direction Change":
            buy_signal = drsi_res['dir_up']
        elif BUY_CONDITION == "Zero-Crossing":
            buy_signal = drsi_res['cross_up']
        else:
            buy_signal = drsi_res['cross_sig_up']

        if SELL_CONDITION == "Direction Change":
            sell_signal = drsi_res['dir_dw']
        elif SELL_CONDITION == "Zero-Crossing":
            sell_signal = drsi_res['cross_dw']
        else:
            sell_signal = drsi_res['cross_sig_dw']

        buy_signal  = buy_signal and drsi_res['filter_pass']
        sell_signal = sell_signal and drsi_res['filter_pass']

        logging.info(f"D-RSI: {drsi_res['drsi']:.4f} | Signal: {drsi_res['signal_line']:.4f} | NRMSE: {drsi_res['nrmse']:.4f} | BUY:{buy_signal} SELL:{sell_signal}")

        # HTF trend filter
        if USE_HTF_TREND_FILTER:
            trend = await self.get_htf_trend()
            logging.info(f"HTF Trend: {trend}")
            if buy_signal and trend != 'bull':
                buy_signal = False
                logging.info("BUY blocked by HTF trend")
            if sell_signal and trend != 'bear':
                sell_signal = False
                logging.info("SELL blocked by HTF trend")

        # SMC confluence
        if USE_SMC_CONFLUENCE:
            df  = self.smc.detect_fvg(df, SMC_FVG_LOOKBACK)
            df  = self.smc.detect_order_blocks(df, SMC_OB_LOOKBACK)
            idx = len(df) - 1
            if buy_signal and not self.smc.is_bullish_confluence(df, idx):
                buy_signal = False
                logging.info("BUY blocked by SMC — no bullish confluence")
            if sell_signal and not self.smc.is_bearish_confluence(df, idx):
                sell_signal = False
                logging.info("SELL blocked by SMC — no bearish confluence")

        # Check open contracts
        self.open_contracts = await self.client.get_open_contracts()
        has_position = len(self.open_contracts) > 0

        # Execute trade
        if not has_position and (buy_signal or sell_signal):
            allowed = await self.is_trading_allowed()
            if allowed:
                atr      = self.get_atr(high, low, close, 14)
                sl_pips  = max(MIN_SL_PIPS, min(MAX_SL_PIPS, int(atr / PIP_VALUE * 0.5)))
                stake    = self.calculate_stake(sl_pips)
                direction = "BUY" if buy_signal else "SELL"

                # Duration = 5 candles of M5 = 25 minutes
                # Adjust based on RR target
                duration = int(DESIRED_RR * 5)

                result = await self.client.buy_contract(
                    direction    = direction,
                    amount       = stake,
                    symbol       = SYMBOL,
                    duration     = duration,
                    duration_unit = "m"
                )

                if result:
                    self.trade_count += 1
                    current_price = close[-1]
                    msg = (
                        f"<b>🚀 Sultan Signal #{self.trade_count}</b>\n"
                        f"Direction: <b>{direction} GOLD</b>\n"
                        f"Entry: {current_price:.2f}\n"
                        f"SL: ~{sl_pips} pips\n"
                        f"Stake: ${stake:.2f}\n"
                        f"Contract ID: {result.get('contract_id', 'N/A')}\n"
                        f"D-RSI: {drsi_res['drsi']:.4f}"
                    )
                    logging.info(msg.replace("<b>", "").replace("</b>", ""))
                    self.telegram.send_message(msg)
                else:
                    logging.error("Trade execution failed")

        # Exit signals
        if EXIT_CONDITION == "Direction Change":
            exit_long  = drsi_res['dir_dw']
            exit_short = drsi_res['dir_up']
        elif EXIT_CONDITION == "Zero-Crossing":
            exit_long  = drsi_res['cross_dw']
            exit_short = drsi_res['cross_up']
        else:
            exit_long  = drsi_res['cross_sig_dw']
            exit_short = drsi_res['cross_sig_up']

        if has_position and (exit_long or exit_short):
            for contract in self.open_contracts:
                cid = contract.get("contract_id")
                if cid:
                    result = await self.client.sell_contract(cid)
                    if result:
                        pnl = result.get("sold_for", 0)
                        msg = f"<b>🔴 Trade Closed</b>\nContract: {cid}\nSold for: ${pnl:.2f}"
                        logging.info(msg.replace("<b>", "").replace("</b>", ""))
                        self.telegram.send_message(msg)

    async def run(self):
        logging.info("🧞 Sultan Gold Bot starting...")
        self.telegram.send_message("🤖 <b>Sultan Gold Bot v2.0 started</b>\nDeriv API | XAUUSD | M5")

        while self.is_running:
            try:
                # Connect / reconnect
                await self.client.connect()

                # Get initial balance
                acc = await self.client.get_account_info()
                if acc:
                    self.daily_start_balance = acc.get("balance", 100.0)
                    self.peak_equity         = self.daily_start_balance
                    logging.info(f"💰 Balance: ${self.daily_start_balance:.2f}")

                self.telegram.send_message(f"💰 Balance: <b>${self.daily_start_balance:.2f}</b>")

                last_candle_time = None

                while self.is_running:
                    try:
                        # Get latest candle time
                        candles = await self.client.get_candles(SYMBOL, GRANULARITY, 2)
                        if candles:
                            latest_time = candles[-1]['epoch']
                            if latest_time != last_candle_time:
                                last_candle_time = latest_time
                                logging.info(f"🕯️ New M5 candle: {datetime.utcfromtimestamp(latest_time)}")
                                await self.analyze_and_trade()
                        await asyncio.sleep(10)

                    except asyncio.TimeoutError:
                        logging.warning("Timeout — reconnecting...")
                        break
                    except Exception as e:
                        logging.error(f"Loop error: {e}")
                        await asyncio.sleep(5)

            except Exception as e:
                logging.error(f"Connection error: {e} — retrying in 15s...")
                await asyncio.sleep(15)

        self.telegram.send_message("🛑 Sultan Bot stopped.")

# ================= ENTRY POINT =================

if __name__ == "__main__":
    bot = SultanGoldBot()
    asyncio.run(bot.run())
