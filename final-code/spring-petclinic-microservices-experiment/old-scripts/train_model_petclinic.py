"""
train_model_petclinic.py

This script trains a dual-input neural network for multi-label classification of anti-patterns in Spring PetClinic microservices.
It loads metrics, embeddings, and labels; standardizes/splits data; trains the model; and evaluates on test set.

Prerequisites:
- Files: metrics/metrics_petclinic.csv, embeddings/embeddings_petclinic.pkl, data/labels_petclinic.csv.
- Libraries: torch, numpy, sklearn, pandas.

Usage: python train_model_petclinic.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

class DualNN(nn.Module):
    """
    Dual-input Neural Network for anti-pattern classification.

    Architecture:
    - Metrics branch: Processes structural metrics.
    - Embeddings branch: Processes semantic embeddings.
    - Combined: Concatenates and outputs sigmoid probabilities for 3 labels.

    Args:
        metrics_dim (int): Dimension of metrics input.
        emb_dim (int): Dimension of embeddings input.
        num_labels (int): Number of output labels (default 3).
    """
    def __init__(self, metrics_dim, emb_dim, num_labels=3):
        super().__init__()
        self.metrics_branch = nn.Sequential(
            nn.Linear(metrics_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.BatchNorm1d(128)
        )
        self.emb_branch = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.BatchNorm1d(256)
        )
        self.final = nn.Sequential(
            nn.Linear(128 + 256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_labels), nn.Sigmoid()
        )

    def forward(self, metrics, emb):
        m_out = self.metrics_branch(metrics)
        e_out = self.emb_branch(emb)
        concat = torch.cat((m_out, e_out), dim=1)
        return self.final(concat)

def main():
    """
    Main function to load data, train, evaluate, and save the model.

    Process:
    - Loads and merges input files.
    - Standardizes metrics and splits data (80/20).
    - Trains for 10 epochs with BCE loss.
    - Evaluates using macro-averaged metrics.

    Returns:
        None: Saves model and prints results.
    """
    print("Starting model training...")  # Debug start
    
    # Load input files
    metrics_df = pd.read_csv('metrics/metrics_petclinic.csv')
    with open('embeddings/embeddings_petclinic.pkl', 'rb') as f:
        emb_dict = pickle.load(f)
    labels_df = pd.read_csv('data/labels_petclinic.csv')
    
    # Merge data on 'service' key
    data = metrics_df.merge(labels_df, on='service')
    embeddings = np.array([emb_dict[row['service']] for _, row in data.iterrows()])
    
    # Define metric columns (must match extract_metrics output)
    metric_cols = ['cbo_mean', 'wmc_mean', 'dit_mean', 'noc_mean', 'rfc_mean', 'lcom_mean', 
                   'loc_mean', 'fanin_mean', 'fanout_mean', 'tcc_mean', 'lcc_mean', 'num_classes']
    X_metrics = data[metric_cols].values
    X_emb = embeddings
    y = data[['mega_service', 'crudy_service', 'ambiguous_service']].values  # Multi-label targets
    
    # Standardize metrics to improve model convergence
    scaler = StandardScaler()
    X_metrics = scaler.fit_transform(X_metrics)
    
    # Split data (80/20 for train/test; random_state for reproducibility)
    X_m_train, X_m_test, X_e_train, X_e_test, y_train, y_test = train_test_split(
        X_metrics, X_emb, y, test_size=0.2, random_state=42
    )
    
    # Save splits as .npz for reuse (e.g., in inference)
    np.savez('data/train_data_petclinic.npz', X_m=X_m_train, X_e=X_e_train, y=y_train)
    np.savez('data/test_data_petclinic.npz', X_m=X_m_test, X_e=X_e_test, y=y_test)
    
    # Convert to tensors for PyTorch
    X_m_train = torch.tensor(X_m_train, dtype=torch.float32)
    X_e_train = torch.tensor(X_e_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_m_test = torch.tensor(X_m_test, dtype=torch.float32)
    X_e_test = torch.tensor(X_e_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    print("Train shapes: Metrics", X_m_train.shape, "Emb", X_e_train.shape, "y", y_train.shape)  # Debug load
    
    # Prepare DataLoader (small batch for limited data; drop_last to avoid size 1 batches)
    train_ds = TensorDataset(X_m_train, X_e_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    
    # Initialize model, loss, and optimizer
    model = DualNN(metrics_dim=X_m_train.shape[1], emb_dim=X_e_train.shape[1])
    criterion = nn.BCELoss()  # Suitable for multi-label binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Model initialized.")  # Debug model
    
    # Training loop (10 epochs as per experiment setup)
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_m, batch_e, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_m, batch_e)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/10, Loss: {total_loss / len(train_loader):.4f}")  # Debug epoch
    
    # Evaluate on test set using macro average for imbalanced labels
    model.eval()
    with torch.no_grad():
        outputs = model(X_m_test, X_e_test)
        preds = (outputs > 0.5).float().numpy()
        y_test_np = y_test.numpy()
    
    acc = accuracy_score(y_test_np, preds)
    prec = precision_score(y_test_np, preds, average='macro', zero_division=0)
    rec = recall_score(y_test_np, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test_np, preds, average='macro', zero_division=0)
    print(f"Test Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")  # Debug eval
    
    # Save trained model for inference
    torch.save(model.state_dict(), 'models/anti_pattern_model_petclinic.pth')
    print("Model saved.")  # Debug save
    
    print("Training complete.")  # Debug end

if __name__ == "__main__":
    main()