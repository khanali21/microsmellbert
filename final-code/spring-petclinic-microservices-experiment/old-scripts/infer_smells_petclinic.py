"""
infer_smells_petclinic.py

This script performs inference using the trained dual-input model on Spring PetClinic data to predict anti-pattern probabilities and binary labels.
It generates results CSV, confidence heatmap, and training loss plot (placeholder losses).

Prerequisites:
- Trained model in 'models/anti_pattern_model_petclinic.pth'.
- Test data .npz or full metrics/embeddings for fallback.
- Libraries: torch, numpy, pandas, matplotlib.

Usage: python infer_smells_petclinic.py
"""

import torch
import torch.nn as nn  # For model definition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DualNN(nn.Module):
    """
    Dual-input Neural Network for anti-pattern inference (same as training model).

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
    Main function for model inference and result visualization.

    Process:
    - Loads model and data (test or full fallback).
    - Computes probabilities and binary predictions.
    - Saves results CSV and plots (heatmap, loss curve).

    Returns:
        None: Saves files directly.
    """
    print("Starting inference...")  # Debug start
    
    # Load model with fixed dimensions (match training)
    model = DualNN(metrics_dim=12, emb_dim=768)  # From your data (12 metric cols)
    model.load_state_dict(torch.load('models/anti_pattern_model_petclinic.pth'))
    model.eval()
    
    # Load test data; fallback to full if .npz missing
    try:
        test_data = np.load('data/test_data_petclinic.npz')
        X_m_test = torch.tensor(test_data['X_m'], dtype=torch.float32)
        X_e_test = torch.tensor(test_data['X_e'], dtype=torch.float32)
    except FileNotFoundError:
        print("No test.npz; using all data for inference.")
        # Load full data as fallback
        metrics_df = pd.read_csv('metrics/metrics_petclinic.csv')
        with open('embeddings/embeddings_petclinic.pkl', 'rb') as f:
            emb_dict = pickle.load(f)
        embeddings = np.array([emb_dict[row['service']] for _, row in metrics_df.iterrows()])
        metric_cols = ['cbo_mean', 'wmc_mean', 'dit_mean', 'noc_mean', 'rfc_mean', 'lcom_mean', 
                       'loc_mean', 'fanin_mean', 'fanout_mean', 'tcc_mean', 'lcc_mean', 'num_classes']
        X_m_test = torch.tensor(metrics_df[metric_cols].values, dtype=torch.float32)
        X_e_test = torch.tensor(embeddings, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(X_m_test, X_e_test)
        probs = outputs.numpy()
        preds = (probs > 0.5).astype(int)
    
    # Generate service names from metrics_df or placeholders
    try:
        metrics_df = pd.read_csv('metrics/metrics_petclinic.csv')
        services = metrics_df['service'].tolist()
    except:
        num_samples = len(probs)
        services = [f'test_sample_{i+1}' for i in range(num_samples)]
    
    # Create results DataFrame with probabilities and predictions
    results_df = pd.DataFrame({
        'service': services,
        'mega_prob': probs[:,0], 'crudy_prob': probs[:,1], 'ambiguous_prob': probs[:,2],
        'mega_pred': preds[:,0], 'crudy_pred': preds[:,1], 'ambiguous_pred': preds[:,2]
    })
    
    # Flag high-confidence predictions for manual review
    for idx, row in results_df.iterrows():
        if row['mega_prob'] > 0.7:
            print(f"High conf Mega-Service in {row['service']}: Prob {row['mega_prob']:.2f}. Check LOC/metrics.")
    
    results_df.to_csv('results/predictions_petclinic.csv', index=False)
    print("Predictions saved to results/predictions_petclinic.csv")
    print(results_df)  # For review

    # Plot confidence scores as bar chart
    results_df.plot(kind='bar', x='service', y=['mega_prob', 'crudy_prob', 'ambiguous_prob'])
    plt.title('Confidence Scores for Anti-Pattern Predictions')
    plt.savefig('results/heatmap_predictions_petclinic.png')
    plt.close()
    # Plot training loss curve (using placeholder values from experiment; replace with actual if logged)
    losses = [0.6886, 0.6504, 0.6172, 0.5925, 0.5397, 0.4965, 0.4641, 0.4350, 0.3924, 0.3372]  # Placeholder from chapter
    plt.plot(range(1, 11), losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/training_loss_curve_petclinic.png')
    plt.close()

if __name__ == "__main__":
    main()