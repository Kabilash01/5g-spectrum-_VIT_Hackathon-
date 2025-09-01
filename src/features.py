"""
Feature engineering for spectrum occupancy prediction.
Creates lag features, rolling statistics, and temporal features for ML models.
"""

import numpy as np


def build_channel_dataset(O, w=10):
    """
    Build feature datasets for each channel to predict next-step occupancy.
    
    Args:
        O: Occupancy matrix shape (T, M), values in {0, 1}
        w: Window size for lag features
    
    Returns:
        List of (X_c, y_c) tuples, one per channel:
        - X_c: Feature matrix shape (T-w, feature_dim)
        - y_c: Labels shape (T-w,), 1 = will be FREE next step
    """
    T, M = O.shape
    channel_datasets = []
    
    for c in range(M):
        # Extract this channel's occupancy
        channel_occ = O[:, c]
        
        # Features and labels
        X_features = []
        y_labels = []
        
        # Start from time w (skip initial w steps due to lag features)
        for t in range(w, T-1):  # T-1 because we need O[t+1] for label
            # Feature 1: Lag vector (most recent first)
            lag_features = channel_occ[t:t-w:-1]  # [O[t], O[t-1], ..., O[t-w+1]]
            
            # Feature 2: Rolling mean over last 5 steps
            r5_start = max(0, t-4)
            r5 = np.mean(channel_occ[r5_start:t+1])
            
            # Feature 3: Rolling mean over last 15 steps  
            r15_start = max(0, t-14)
            r15 = np.mean(channel_occ[r15_start:t+1])
            
            # Feature 4: Time-of-minute feature [0, 1]
            time_feature = (t % 60) / 60.0
            
            # Combine all features
            features = np.concatenate([lag_features, [r5, r15, time_feature]])
            X_features.append(features)
            
            # Label: 1 if channel will be FREE at next step
            y_free = 1 - O[t+1, c]  # Convert busy->free: 0->1, 1->0
            y_labels.append(y_free)
        
        # Convert to numpy arrays
        X_c = np.array(X_features)
        y_c = np.array(y_labels)
        
        channel_datasets.append((X_c, y_c))
    
    return channel_datasets


def get_feature_names(w=10):
    """
    Get human-readable feature names for interpretability.
    
    Args:
        w: Window size for lag features
    
    Returns:
        List of feature names
    """
    feature_names = []
    
    # Lag features
    for i in range(w):
        feature_names.append(f'lag_{i}')
    
    # Rolling statistics
    feature_names.extend(['rolling_5', 'rolling_15', 'time_of_minute'])
    
    return feature_names


def print_dataset_summary(channel_datasets):
    """
    Print summary statistics for the channel datasets.
    
    Args:
        channel_datasets: List of (X_c, y_c) tuples from build_channel_dataset
    """
    print(f"\nDataset Summary:")
    print(f"Number of channels: {len(channel_datasets)}")
    
    for c, (X_c, y_c) in enumerate(channel_datasets):
        n_samples = len(X_c)
        n_features = X_c.shape[1] if n_samples > 0 else 0
        n_free = np.sum(y_c) if n_samples > 0 else 0
        n_busy = n_samples - n_free
        
        print(f"Channel {c}: {n_samples} samples, {n_features} features")
        print(f"  - Will be FREE: {n_free} ({n_free/n_samples*100:.1f}%)")
        print(f"  - Will be BUSY: {n_busy} ({n_busy/n_samples*100:.1f}%)")


if __name__ == "__main__":
    # Test with simulated data
    from simulate import simulate_occupancy
    
    print("Testing feature engineering...")
    
    # Generate test data
    O = simulate_occupancy(T=100, M=4, seed=42)
    
    # Build features
    w = 5
    datasets = build_channel_dataset(O, w=w)
    
    # Print summary
    print_dataset_summary(datasets)
    
    # Show feature names
    feature_names = get_feature_names(w=w)
    print(f"\nFeature names: {feature_names}")
    
    # Sample features for first channel
    if len(datasets) > 0:
        X_0, y_0 = datasets[0]
        if len(X_0) > 0:
            print(f"\nSample features (first 3 rows of channel 0):")
            for i in range(min(3, len(X_0))):
                print(f"Row {i}: {X_0[i]}")
                print(f"Label {i}: {y_0[i]} (will be {'FREE' if y_0[i] else 'BUSY'})")
