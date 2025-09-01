"""
Machine learning forecasting for spectrum availability.
Trains per-channel LogisticRegression models to predict next-step channel occupancy.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def train_per_channel_models(channel_datasets, max_iter=200):
    """
    Train LogisticRegression model for each channel to predict P(free at next step).
    
    Args:
        channel_datasets: List of (X_c, y_c) tuples from build_channel_dataset
        max_iter: Maximum iterations for LogisticRegression
    
    Returns:
        List of trained models (or None for channels with insufficient data)
    """
    models = []
    
    for c, (X_c, y_c) in enumerate(channel_datasets):
        # Check if we have enough data and both classes
        if len(X_c) < 10:
            print(f"Channel {c}: Insufficient data ({len(X_c)} samples), using heuristic")
            models.append(None)
            continue
        
        unique_labels = np.unique(y_c)
        if len(unique_labels) < 2:
            print(f"Channel {c}: Only one class present ({unique_labels}), using heuristic")
            models.append(None)
            continue
        
        # Train LogisticRegression
        try:
            model = LogisticRegression(max_iter=max_iter, random_state=42)
            model.fit(X_c, y_c)
            
            # Quick validation
            train_accuracy = model.score(X_c, y_c)
            print(f"Channel {c}: Trained successfully, accuracy={train_accuracy:.3f}")
            
            models.append(model)
            
        except Exception as e:
            print(f"Channel {c}: Training failed ({e}), using heuristic")
            models.append(None)
    
    return models


def predict_p_free(models, O, w=10):
    """
    Predict probability that each channel will be free at next time step.
    
    Args:
        models: List of trained models from train_per_channel_models
        O: Occupancy matrix shape (T, M)
        w: Window size for lag features (must match training)
    
    Returns:
        np.ndarray shape (T, M): Predicted P(free) for each channel at each time
    """
    T, M = O.shape
    p_free = np.zeros((T, M))
    
    # For times before w, use heuristic (not enough history)
    for t in range(w):
        for c in range(M):
            # Simple heuristic: recent average availability
            recent_window = max(1, t+1)
            recent_occ = O[max(0, t-recent_window+1):t+1, c]
            p_free[t, c] = 1 - np.mean(recent_occ)
    
    # For times >= w, use models or heuristics
    for t in range(w, T):
        for c in range(M):
            # Build features for this time step (same as in features.py)
            channel_occ = O[:, c]
            
            # Lag features (most recent first)
            lag_features = channel_occ[t:t-w:-1]
            
            # Rolling means
            r5_start = max(0, t-4)
            r5 = np.mean(channel_occ[r5_start:t+1])
            
            r15_start = max(0, t-14)
            r15 = np.mean(channel_occ[r15_start:t+1])
            
            # Time feature
            time_feature = (t % 60) / 60.0
            
            # Combine features
            features = np.concatenate([lag_features, [r5, r15, time_feature]])
            features = features.reshape(1, -1)  # Shape for single prediction
            
            # Make prediction
            if models[c] is not None:
                # Use trained model
                p_free[t, c] = models[c].predict_proba(features)[0, 1]  # P(class=1) = P(free)
            else:
                # Heuristic fallback: complement of recent rolling mean
                p_free[t, c] = 1 - r5
    
    # Clip probabilities to reasonable range
    p_free = np.clip(p_free, 0.01, 0.99)
    
    return p_free


def evaluate_predictions(models, channel_datasets):
    """
    Evaluate prediction accuracy on training data.
    
    Args:
        models: Trained models
        channel_datasets: Training datasets
    
    Returns:
        Dict with evaluation metrics
    """
    results = {
        'channel_accuracies': [],
        'overall_accuracy': 0.0,
        'n_models_trained': 0,
        'n_heuristic_fallbacks': 0
    }
    
    all_predictions = []
    all_labels = []
    
    for c, (X_c, y_c) in enumerate(channel_datasets):
        if models[c] is not None and len(X_c) > 0:
            # Model predictions
            y_pred = models[c].predict(X_c)
            accuracy = np.mean(y_pred == y_c)
            results['channel_accuracies'].append(accuracy)
            results['n_models_trained'] += 1
            
            all_predictions.extend(y_pred)
            all_labels.extend(y_c)
        else:
            # Heuristic fallback
            results['n_heuristic_fallbacks'] += 1
    
    if len(all_predictions) > 0:
        results['overall_accuracy'] = np.mean(np.array(all_predictions) == np.array(all_labels))
    
    return results


if __name__ == "__main__":
    # Test with simulated data
    from simulate import simulate_occupancy
    from features import build_channel_dataset
    
    print("Testing ML forecasting...")
    
    # Generate test data
    O = simulate_occupancy(T=200, M=4, seed=42)
    
    # Build features
    w = 10
    datasets = build_channel_dataset(O, w=w)
    
    # Train models
    print("\nTraining models...")
    models = train_per_channel_models(datasets)
    
    # Evaluate
    results = evaluate_predictions(models, datasets)
    print(f"\nEvaluation Results:")
    print(f"Models trained: {results['n_models_trained']}")
    print(f"Heuristic fallbacks: {results['n_heuristic_fallbacks']}")
    print(f"Overall accuracy: {results['overall_accuracy']:.3f}")
    
    # Test prediction
    print("\nTesting prediction...")
    p_free = predict_p_free(models, O, w=w)
    print(f"Predicted probabilities shape: {p_free.shape}")
    print(f"Probability range: [{p_free.min():.3f}, {p_free.max():.3f}]")
    print(f"Mean probability: {p_free.mean():.3f}")
