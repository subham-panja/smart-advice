# scripts/predictor.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scripts.data_fetcher import get_historical_data
from scripts.deep_learning_models import LSTMModel
from utils.logger import setup_logging
logger = setup_logging()

class PricePredictor:
    """
    Manages the training and execution of deep learning models for price prediction.
    """
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

    def _create_sequences(self, input_data, tw):
        """
        Create sequences for training the LSTM model.
        
        Args:
            input_data: The input time series data.
            tw (int): The sequence length (time window).
            
        Returns:
            A tuple of sequences and their corresponding labels.
        """
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + tw:i + tw + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    def prepare_data(self, time_window=12):
        """
        Prepares data for model training and prediction.
        
        Args:
            time_window (int): The number of past periods to use for prediction.
            
        Returns:
            A tuple of training data sequences and the scaled test data.
        """
        data = get_historical_data(self.symbol, period="2y")
        if data is None or data.empty:
            logger.warning(f"No data for {self.symbol}, cannot prepare for prediction.")
            return None, None
        
        close_prices = data['Close'].values.astype(float)
        
        # Normalize the data
        test_data_size = time_window
        train_data = close_prices[:-test_data_size]
        test_data = close_prices[-test_data_size:]
        
        train_data_normalized = self.scaler.fit_transform(train_data.reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

        # Create sequences for training
        train_inout_seq = self._create_sequences(train_data_normalized, time_window)
        
        return train_inout_seq, test_data

    def train(self, train_inout_seq, epochs=150):
        """
        Trains the LSTM model.
        
        Args:
            train_inout_seq: The training data sequences.
            epochs (int): The number of training epochs.
        """
        if not train_inout_seq:
            logger.warning(f"No training data for {self.symbol}, skipping training.")
            return

        self.model = LSTMModel(input_size=1, hidden_layer_size=100)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        logger.info(f"Starting model training for {self.symbol} for {epochs} epochs...")
        for i in range(epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                        torch.zeros(1, 1, self.model.hidden_layer_size))

                y_pred = self.model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if (i + 1) % 25 == 0:
                logger.debug(f'Epoch {i+1}/{epochs} loss: {single_loss.item()}')
        
        logger.info(f"Model training for {self.symbol} complete.")

    def predict_next_day_price(self, test_data, time_window=12):
        """
        Predicts the next day's closing price.
        
        Args:
            test_data: The recent historical data to use for prediction.
            time_window (int): The number of past periods to use for prediction.

        Returns:
            The predicted closing price for the next day.
        """
        if self.model is None:
            logger.warning(f"Model for {self.symbol} is not trained. Cannot predict.")
            return None

        fut_pred = time_window
        test_inputs = list(test_data[-time_window:]) # Ensure it's a list
        
        self.model.eval()
        
        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[-time_window:])
            with torch.no_grad():
                self.model.hidden = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                   torch.zeros(1, 1, self.model.hidden_layer_size))
                test_inputs.append(self.model(seq).item())
        
        # Inverse transform the prediction to get the actual price
        actual_predictions = self.scaler.inverse_transform(np.array(test_inputs[time_window:]).reshape(-1, 1))
        
        return actual_predictions[0][0]

if __name__ == '__main__':
    predictor = PricePredictor('RELIANCE.NS')
    train_data, test_data = predictor.prepare_data()
    if train_data and test_data is not None:
        predictor.train(train_data)
        predicted_price = predictor.predict_next_day_price(test_data)
        if predicted_price is not None:
            print(f"\n=== Price Prediction Example ===")
            print(f"Predicted next day's closing price for RELIANCE.NS: {predicted_price:.2f}")

