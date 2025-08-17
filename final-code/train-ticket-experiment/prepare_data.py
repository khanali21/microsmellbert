import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    print("Starting data preparation...")  # Debug start
    
    # Load labels
    labels_df = pd.read_csv('labels.csv')
    print("Labels shape:", labels_df.shape)  # Debug labels
    print("Labels head:\n", labels_df.head())
    
    # Load metrics (aggregate features)
    metrics_df = pd.read_csv('metrics/metrics.csv')
    print("Metrics shape:", metrics_df.shape)  # Debug metrics
    print("Metrics head:\n", metrics_df.head())
    
    # Load embeddings (flatten if needed)
    emb_df = pd.read_pickle('embeddings/embeddings.pkl')
    emb_df['embedding'] = emb_df['embedding'].apply(lambda x: np.array(x).flatten())  # Ensure flat
    print("Embeddings shape:", emb_df.shape)  # Debug emb
    print("Embeddings head:\n", emb_df.head())
    
    # Merge on 'service' (inner join; drops unmatched)
    merged = labels_df.merge(metrics_df, on='service', how='inner')
    merged = merged.merge(emb_df, on='service', how='inner')
    print("Merged shape:", merged.shape)  # Debug merged
    
    # Features: Metrics cols (exclude 'service'), embeddings as array
    metric_cols = [col for col in metrics_df.columns if col != 'service']
    X_metrics = merged[metric_cols].values
    X_emb = np.stack(merged['embedding'].values)  # Stack to array (n_samples, 768)
    y = merged[['mega_service', 'crudy_service', 'ambiguous_service']].values  # Multi-label
    
    # Scale metrics (standardize)
    scaler = StandardScaler()
    X_metrics = scaler.fit_transform(X_metrics)
    print("Scaled X_metrics shape:", X_metrics.shape)  # Debug scale
    
    # Split: 80/20 (small data; use cross-val in full Capstone)
    X_m_train, X_m_test, X_e_train, X_e_test, y_train, y_test = train_test_split(
        X_metrics, X_emb, y, test_size=0.2, random_state=42
    )
    print("Train shapes: Metrics", X_m_train.shape, "Emb", X_e_train.shape, "y", y_train.shape)  # Debug split
    
    # Save for model (numpy for PyTorch)
    np.savez('data/train_data.npz', X_m=X_m_train, X_e=X_e_train, y=y_train)
    np.savez('data/test_data.npz', X_m=X_m_test, X_e=X_e_test, y=y_test)
    print("Data saved as .npz files.")  # Debug save
    
    print("Data preparation complete.")  # Debug end

if __name__ == "__main__":
    main()