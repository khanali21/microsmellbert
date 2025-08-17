import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class DualNN(nn.Module):
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
    print("Starting model training...")  # Debug start
    
    # Load data
    train_data = np.load('data/train_data.npz')
    test_data = np.load('data/test_data.npz')
    X_m_train, X_e_train, y_train = torch.tensor(train_data['X_m'], dtype=torch.float32), torch.tensor(train_data['X_e'], dtype=torch.float32), torch.tensor(train_data['y'], dtype=torch.float32)
    X_m_test, X_e_test, y_test = torch.tensor(test_data['X_m'], dtype=torch.float32), torch.tensor(test_data['X_e'], dtype=torch.float32), torch.tensor(test_data['y'], dtype=torch.float32)
    print("Train shapes: Metrics", X_m_train.shape, "Emb", X_e_train.shape, "y", y_train.shape)  # Debug load
    
    # DataLoader (batch=8 for small data, drop_last=True to avoid batch size 1)
    train_ds = TensorDataset(X_m_train, X_e_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    
    # Model (adjust dims based on your data: metrics_dim = num metric cols, emb_dim=768)
    model = DualNN(metrics_dim=X_m_train.shape[1], emb_dim=X_e_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Model initialized.")  # Debug model
    
    # Train (10 epochs; small data)
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
    
    # Evaluate on test
    model.eval()
    with torch.no_grad():
        outputs = model(X_m_test, X_e_test)
        preds = (outputs > 0.5).float().numpy()
        y_test_np = y_test.numpy()
    
    # Metrics (multi-label: average='macro' for balance)
    acc = accuracy_score(y_test_np, preds)
    prec = precision_score(y_test_np, preds, average='macro', zero_division=0)
    rec = recall_score(y_test_np, preds, average='macro', zero_division=0)
    f1 = f1_score(y_test_np, preds, average='macro', zero_division=0)
    print(f"Test Metrics: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")  # Debug eval
    
    # Save model
    torch.save(model.state_dict(), 'models/anti_pattern_model.pth')
    print("Model saved.")  # Debug save
    
    print("Training complete.")  # Debug end

if __name__ == "__main__":
    main()