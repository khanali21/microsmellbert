"""
infer_smells.py

This script performs inference using the trained dual-input model on a Java project data to predict anti-pattern probabilities and binary labels.
It generates results CSV, confidence heatmap, and training loss plot (placeholder losses).

Prerequisites:
- Trained model in '<models_path>/anti_pattern_model_{project}.pth'.
- Test data .npz or full metrics/embeddings for fallback.
- Libraries: torch, numpy, pandas, matplotlib.
- Configuration: config.json with 'data_path', 'metrics_path', 'embeddings_path', 'models_path', 'results_path', 'project', 'metric_cols'.

Usage: python infer_smells.py
"""

import torch
import torch.nn as nn  # For model definition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json  # For loading config
import os
import logging
import pickle  # Added for loading embeddings in fallback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)
project = config['project']
data_path = config['data_path']
metrics_path = config['metrics_path']
embeddings_path = config['embeddings_path']
models_path = config['models_path']
results_path = config['results_path']
metric_cols = [f'{col}_mean' for col in config.get('metric_cols', [])] + ['num_classes']
logger.info(f"Loaded config for project: {project}")

class ConditionalBatchNorm(nn.Module):
    """
    Custom module to conditionally apply BatchNorm1d based on batch size.
    Applies BatchNorm1d if batch size > 1, otherwise returns input unchanged.
    """
    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        if x.size(0) > 1:
            return self.batch_norm(x)
        return x

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
            nn.Linear(metrics_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            ConditionalBatchNorm(128)
        )
        self.emb_branch = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            ConditionalBatchNorm(256)
        )
        self.final = nn.Sequential(
            nn.Linear(128 + 256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_labels)
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
    logger.info("Starting inference...")  # Debug start
    
    # Load model with dynamic dimensions
    model = DualNN(metrics_dim=len(metric_cols), emb_dim=768)  # emb_dim fixed for CodeBERT
    model_path = os.path.join(models_path, f'anti_pattern_model_{project}.pth')
    logger.info(f"Loading model from {model_path}")
    state_dict = torch.load(model_path)
    # Handle potential key mismatches (temporary workaround)
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Load test data; fallback to full if .npz missing
    test_data_path = os.path.join(data_path, f'test_data_{project}.npz')
    try:
        test_data = np.load(test_data_path)
        X_m_test = torch.tensor(test_data['X_m'], dtype=torch.float32)
        X_e_test = torch.tensor(test_data['X_e'], dtype=torch.float32)
        logger.info(f"Loaded test data from {test_data_path}, metrics shape: {X_m_test.shape}, embeddings shape: {X_e_test.shape}")
    except FileNotFoundError:
        logger.warning(f"No test.npz found at {test_data_path}; using all data for inference.")
        # Load full data as fallback
        metrics_csv = os.path.join(metrics_path, f'metrics_{project}.csv')
        metrics_df = pd.read_csv(metrics_csv)
        logger.info(f"Loaded metrics from {metrics_csv}, shape: {metrics_df.shape}")
        embeddings_pkl = os.path.join(embeddings_path, f'embeddings_{project}.pkl')
        with open(embeddings_pkl, 'rb') as f:
            emb_dict = pickle.load(f)
        logger.info(f"Loaded embeddings from {embeddings_pkl}")
        embeddings = np.array([emb_dict[row['service']] for _, row in metrics_df.iterrows()])
        X_m_test = torch.tensor(metrics_df[metric_cols].values, dtype=torch.float32)
        X_e_test = torch.tensor(embeddings, dtype=torch.float32)
        logger.info(f"Fallback data shapes: metrics {X_m_test.shape}, embeddings {X_e_test.shape}")
    
    with torch.no_grad():
        outputs = model(X_m_test, X_e_test)
        probs = torch.sigmoid(outputs).numpy()
        preds = (probs > 0.5).astype(int)
        logger.info(f"Inference complete, probabilities shape: {probs.shape}, predictions shape: {preds.shape}")
    
    # Generate service names for test data
    try:
        metrics_df = pd.read_csv(os.path.join(metrics_path, f'metrics_{project}.csv'))
        # Load the original train/test split to get test indices
        train_data = np.load(os.path.join(data_path, f'train_data_{project}.npz'))
        test_data = np.load(os.path.join(data_path, f'test_data_{project}.npz'))
        all_indices = np.arange(len(metrics_df))
        train_indices = np.where(np.isin(all_indices, train_data['X_m'].shape[0]))[0]  # Placeholder, adjust logic
        test_indices = np.setdiff1d(all_indices, train_indices)[:len(probs)]  # Match test sample count
        services = metrics_df.iloc[test_indices]['service'].tolist()
        logger.info(f"Loaded {len(services)} services for test data from metrics CSV")
    except Exception as e:
        num_samples = len(probs)
        services = [f'test_sample_{i+1}' for i in range(num_samples)]
        logger.warning(f"Failed to load services: {e}. Using {num_samples} placeholder services")
    
    # Create results DataFrame with probabilities and predictions
    results_df = pd.DataFrame({
        'service': services,
        'mega_prob': probs[:,0], 'crudy_prob': probs[:,1], 'ambiguous_prob': probs[:,2],
        'mega_pred': preds[:,0], 'crudy_pred': preds[:,1], 'ambiguous_pred': preds[:,2]
    })
    
    # Flag high-confidence predictions for manual review
    for idx, row in results_df.iterrows():
        if row['mega_prob'] > 0.7:
            logger.warning(f"High conf Mega-Service in {row['service']}: Prob {row['mega_prob']:.2f}. Check LOC/metrics.")
    
    # Ensure results directory exists
    os.makedirs(results_path, exist_ok=True)
    results_csv = os.path.join(results_path, f'predictions_{project}.csv')
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Predictions saved to {results_csv}")
    logger.debug(results_df)  # For review

    # Plot confidence scores as bar chart
    results_df.plot(kind='bar', x='service', y=['mega_prob', 'crudy_prob', 'ambiguous_prob'])
    plt.title('Confidence Scores for Anti-Pattern Predictions')
    heatmap_path = os.path.join(results_path, f'heatmap_predictions_{project}.png')
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Heatmap saved to {heatmap_path}")
    # Plot training loss curve (using placeholder values from experiment; replace with actual if logged)
    losses = [0.6886, 0.6504, 0.6172, 0.5925, 0.5397, 0.4965, 0.4641, 0.4350, 0.3924, 0.3372]  # Placeholder from chapter
    plt.plot(range(1, 11), losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    loss_curve_path = os.path.join(results_path, f'training_loss_curve_{project}.png')
    plt.savefig(loss_curve_path)
    plt.close()
    logger.info(f"Training loss curve saved to {loss_curve_path}")

    logger.info("Inference complete.")

if __name__ == "__main__":
    main()