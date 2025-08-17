import torch
import torch.nn as nn  # Added missing import for nn
import numpy as np
import pandas as pd

# DualNN class (copy from train_model.py if needed)
class DualNN(torch.nn.Module):
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
    print("Starting inference...")  # Debug start
    
    # Load model
    model = DualNN(metrics_dim=12, emb_dim=768)  # From your data (12 metric cols)
    model.load_state_dict(torch.load('models/anti_pattern_model.pth'))
    model.eval()
    
    # Load test data (add service names for explanations; assume from labels.csv test split)
    test_data = np.load('data/test_data.npz')
    X_m_test = torch.tensor(test_data['X_m'], dtype=torch.float32)
    X_e_test = torch.tensor(test_data['X_e'], dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(X_m_test, X_e_test)
        probs = outputs.numpy()
        preds = (probs > 0.5).astype(int)
    
    # Generate placeholder service names dynamically based on number of test samples
    num_samples = len(probs)
    services = [f'test_sample_{i+1}' for i in range(num_samples)]  # e.g., 'test_sample_1', etc.
    
    # Results DF with confidence
    results_df = pd.DataFrame({
        'service': services,
        'mega_prob': probs[:,0], 'crudy_prob': probs[:,1], 'ambiguous_prob': probs[:,2],
        'mega_pred': preds[:,0], 'crudy_pred': preds[:,1], 'ambiguous_pred': preds[:,2]
    })
    
    # Explanatory: Flag high-conf with snippets (placeholder; link to code)
    for idx, row in results_df.iterrows():
        if row['mega_prob'] > 0.7:
            print(f"High conf Mega-Service in {row['service']}: Prob {row['mega_prob']:.2f}. Check LOC/metrics.")
    
    results_df.to_csv('results/predictions.csv', index=False)
    print("Predictions saved to results/predictions.csv")
    print(results_df)  # For review

    import matplotlib.pyplot as plt
    results_df.plot(kind='bar', x='service', y=['mega_prob', 'crudy_prob', 'ambiguous_prob'])
    plt.title('Confidence Scores for Anti-Pattern Predictions')
    plt.savefig('results/heatmap_predictions.png')
    plt.close()
    # Loss plot (manual for now; use logged losses)
    losses = [0.6886, 0.6504, 0.6172, 0.5925, 0.5397, 0.4965, 0.4641, 0.4350, 0.3924, 0.3372]
    plt.plot(range(1, 11), losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/training_loss_curve.png')
    plt.close()

if __name__ == "__main__":
    main()