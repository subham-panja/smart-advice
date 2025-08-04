# scripts/rl_trading_agent.py

import gymnasium as gym
import numpy as np
import pandas as pd
import talib as ta
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from scripts.data_fetcher import get_historical_data
from utils.logger import setup_logging
from typing import Dict, Any
logger = setup_logging()

class StockTradingEnv(gym.Env):
    """
    A stock trading environment for reinforcement learning.
    
    This environment simulates stock trading, allowing an RL agent to learn
    trading strategies by interacting with historical market data.
    """
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.reward_range = (-np.inf, np.inf)
        
        # Actions: 0 -> Hold, 1 -> Buy, 2 -> Sell
        self.action_space = spaces.Discrete(3)
        
        # Observations: Price data and other features
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, len(df.columns)), dtype=np.float16
        )
        
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        frame = np.array([
            self.df.iloc[self.current_step:self.current_step + 6]
        ])
        return frame

    def step(self, action):
        self.current_step += 1
        
        # Simplified reward logic
        if action == 1: # Buy
            reward = self.df['Close'].iloc[self.current_step] - self.df['Open'].iloc[self.current_step]
        elif action == 2: # Sell
            reward = self.df['Open'].iloc[self.current_step] - self.df['Close'].iloc[self.current_step]
        else: # Hold
            reward = 0
            
        done = self.current_step >= len(self.df) - 7
        obs = self._next_observation()
        
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        pass

class RLTradingAgent:
    """
    A Reinforcement Learning agent for making trading decisions.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None

    def run_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs a simplified, heuristic-based analysis for quick insights without full RL training.

        Args:
            df (pd.DataFrame): Historical data for the stock.

        Returns:
            Dict[str, Any]: A dictionary containing the trading action and reasoning.
        """
        logger.info(f"Running simplified RL agent analysis for {self.symbol}")

        # Basic heuristic: Check momentum and mean reversion signals
        latest_price = df['Close'].iloc[-1]
        ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
        ma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
        rsi = ta.RSI(df['Close'], timeperiod=14).iloc[-1]

        action = 'HOLD'
        reason = "Default action: No strong signal detected."

        # Momentum signal
        if latest_price > ma_20 and ma_20 > ma_50:
            action = 'BUY'
            reason = "Strong upward momentum detected (Price > MA20 > MA50)."

        # Mean reversion signal
        elif rsi < 30:
            action = 'BUY'
            reason = f"Potential mean reversion opportunity (RSI is oversold at {rsi:.2f})."

        elif latest_price < ma_20 and ma_20 < ma_50:
            action = 'SELL'
            reason = "Strong downward momentum detected (Price < MA20 < MA50)."
        
        elif rsi > 70:
            action = 'SELL'
            reason = f"Potential mean reversion opportunity (RSI is overbought at {rsi:.2f})."

        return {
            'action': action,
            'action_reason': reason,
            'details': {
                'rsi': rsi,
                'ma_20': ma_20,
                'ma_50': ma_50
            }
        }

    def train(self, total_timesteps=10000):
        """
        Trains the RL trading agent.
        
        Args:
            total_timesteps (int): The total number of training steps.
        """
        data = get_historical_data(self.symbol, period="5y")
        if data is None or len(data) < 20: # Need enough data to train
            logger.warning(f"Not enough data for {self.symbol}, skipping RL training.")
            return
            
        # Preprocess data
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change().dropna()
        
        env = DummyVecEnv([lambda: StockTradingEnv(data)])
        self.model = PPO('MlpPolicy', env, verbose=0)
        
        logger.info(f"Starting RL model training for {self.symbol}...")
        self.model.learn(total_timesteps=total_timesteps)
        logger.info(f"RL model training for {self.symbol} complete.")

    def predict_action(self, df):
        """
        Predicts the next trading action (Buy, Sell, or Hold).
        
        Args:
            df: The recent historical data to use for prediction.

        Returns:
            The predicted action (0: Hold, 1: Buy, 2: Sell).
        """
        if self.model is None:
            logger.warning(f"RL model for {self.symbol} is not trained. Cannot predict.")
            return 0 # Default to Hold

        # Prepare the observation
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change().dropna()
        obs = np.array([df.tail(6)])
        
        action, _ = self.model.predict(obs)
        return action[0]

if __name__ == '__main__':
    agent = RLTradingAgent('RELIANCE.NS')
    agent.train(total_timesteps=20000) # Use more timesteps for real training
    
    # Get recent data for prediction
    data_for_pred = get_historical_data('RELIANCE.NS', period="1mo")
    if data_for_pred is not None:
        predicted_action = agent.predict_action(data_for_pred)
        action_map = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        
        print(f"\n=== RL Trading Agent Example ===")
        print(f"Predicted action for RELIANCE.NS: {action_map[predicted_action]}")

