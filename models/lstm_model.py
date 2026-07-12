"""
Modele LSTM (PyTorch) pour la prediction de series temporelles financieres.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import joblib

MODELS_DIR = Path(__file__).parent / "saved"

SEQUENCE_LENGTH = 60
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001


class LSTMNetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


def create_sequences(data: np.ndarray, seq_length: int) -> tuple:
    """Cree des sequences pour le LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def prepare_data(df: pd.DataFrame, ticker: str) -> tuple:
    """Prepare les donnees pour le LSTM."""
    data = df[df["Ticker"] == ticker]["Close"].values.reshape(-1, 1)

    if len(data) < SEQUENCE_LENGTH + 100:
        return None, None, None, None, None

    # Normalisation
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Creation des sequences
    X, y = create_sequences(data_scaled, SEQUENCE_LENGTH)

    # Split temporel (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, scaler


def train_lstm(df: pd.DataFrame, ticker: str, epochs: int = EPOCHS) -> dict:
    """Entraine un modele LSTM sur un ticker."""
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, ticker)

    if X_train is None:
        print(f"  [!] Pas assez de donnees pour {ticker}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Conversion en tenseurs
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Modele
    model = LSTMNetwork(input_size=1, hidden_size=64, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Entrainement
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"    Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy()
        y_test_actual = scaler.inverse_transform(y_test)
        y_pred_actual = scaler.inverse_transform(y_pred_scaled)

    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100

    return {
        "model": model,
        "scaler": scaler,
        "ticker": ticker,
        "metrics": {"mae": float(mae), "mape": float(mape)},
        "train_size": len(X_train),
        "test_size": len(X_test),
        "device": str(device),
    }


def save_model(result: dict) -> Path:
    """Sauvegarde le modele LSTM et son scaler."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ticker = result["ticker"]
    base_name = f"lstm_{ticker.replace('^', 'IDX_')}"

    model_path = MODELS_DIR / f"{base_name}.pt"
    scaler_path = MODELS_DIR / f"{base_name}_scaler.pkl"

    torch.save(result["model"].state_dict(), model_path)
    joblib.dump(result["scaler"], scaler_path)
    return model_path


def load_model(ticker: str) -> tuple:
    """Charge un modele LSTM et son scaler."""
    base_name = f"lstm_{ticker.replace('^', 'IDX_')}"
    model = LSTMNetwork(input_size=1, hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load(MODELS_DIR / f"{base_name}.pt", weights_only=True))
    model.eval()
    scaler = joblib.load(MODELS_DIR / f"{base_name}_scaler.pkl")
    return model, scaler
