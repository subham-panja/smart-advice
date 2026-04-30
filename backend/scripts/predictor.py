import logging

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from scripts.data_fetcher import get_historical_data
from scripts.deep_learning_models import LSTMModel

logger = logging.getLogger(__name__)


class PricePredictor:
    """
    Manages training and execution of LSTM models with strict data requirements.
    """

    def __init__(self, symbol: str, app_config: Dict):
        self.symbol = symbol
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        # Mandatory window from config
        self.time_window = app_config["prediction_config"]["time_window"]
        self.epochs = app_config["prediction_config"]["epochs"]

    def prepare_data(self) -> Tuple[List, np.ndarray]:
        """Prepares data strictly."""
        data = get_historical_data(self.symbol, period="2y")
        if data.empty:
            raise ValueError(f"Insufficient data for {self.symbol} prediction.")

        close_prices = data["Close"].values.astype(float)
        test_data_size = self.time_window
        train_data = close_prices[:-test_data_size]
        test_data = close_prices[-test_data_size:]

        train_data_normalized = self.scaler.fit_transform(train_data.reshape(-1, 1))
        train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

        train_inout_seq = []
        L = len(train_data_normalized)
        for i in range(L - self.time_window):
            train_seq = train_data_normalized[i : i + self.time_window]
            train_label = train_data_normalized[i + self.time_window : i + self.time_window + 1]
            train_inout_seq.append((train_seq, train_label))

        return train_inout_seq, test_data

    def train(self, train_inout_seq: List):
        """Trains the LSTM model with no silent step failures."""
        if not train_inout_seq:
            raise ValueError(f"No training sequences generated for {self.symbol}.")

        try:
            self.model = LSTMModel(input_size=1, hidden_layer_size=50)
            loss_function = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            logger.info(f"Starting strict model training for {self.symbol}...")
            for i in range(self.epochs):
                for seq, labels in train_inout_seq:
                    optimizer.zero_grad()
                    seq = seq.view(1, -1)
                    labels = labels.view(1, -1)
                    y_pred = self.model(seq)
                    single_loss = loss_function(y_pred, labels)
                    single_loss.backward()
                    optimizer.step()
            logger.info(f"Model training for {self.symbol} complete.")
        except Exception as e:
            logger.error(f"Critical training failure for {self.symbol}: {e}")
            raise e

    def predict_next_day_price(self, test_data: np.ndarray) -> float:
        """Predicts price with no fallbacks."""
        if self.model is None:
            raise RuntimeError(f"Model for {self.symbol} is not trained.")

        try:
            self.model.eval()
            normalized_test_data = self.scaler.transform(test_data.reshape(-1, 1))
            test_inputs = torch.FloatTensor(normalized_test_data[-self.time_window :]).view(1, -1)

            with torch.no_grad():
                prediction = self.model(test_inputs)

            predicted_price = self.scaler.inverse_transform(prediction.numpy())
            return float(predicted_price[0][0])
        except Exception as e:
            logger.error(f"Prediction failure for {self.symbol}: {e}")
            raise e
