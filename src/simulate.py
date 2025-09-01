"""
Spectrum occupancy and user demand simulators.
Generates realistic radio spectrum usage patterns without hardware.
"""

import numpy as np
import os


def simulate_occupancy(T=600, M=8, seed=42):
    """
    Simulate spectrum occupancy over time with realistic patterns.
    
    Args:
        T: Number of time steps
        M: Number of channels
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray[int] shape (T, M): Occupancy matrix, 1=busy, 0=free
    """
    np.random.seed(seed)
    
    # Initialize occupancy matrix
    O = np.zeros((T, M), dtype=int)
    
    # Per-channel parameters
    base_busy_prob = np.random.uniform(0.2, 0.6, M)  # Base busy probability
    persistence = np.random.uniform(0.7, 0.95, M)    # Markov persistence
    seasonality_phase = np.random.uniform(0, 2*np.pi, M)  # Phase offset for seasonality
    seasonality_amplitude = np.random.uniform(0.1, 0.3, M)  # Seasonality strength
    
    # Initialize first time step
    O[0, :] = (np.random.random(M) < base_busy_prob).astype(int)
    
    for t in range(1, T):
        for c in range(M):
            # Base probability for this channel
            p_base = base_busy_prob[c]
            
            # Persistence: if busy before, more likely to stay busy
            if O[t-1, c] == 1:
                p_persist = persistence[c]
            else:
                p_persist = 1 - persistence[c]
            
            # Seasonality: sinusoidal pattern over 60-step period
            seasonality = seasonality_amplitude[c] * np.sin(2 * np.pi * t / 60 + seasonality_phase[c])
            
            # Occasional burst noise (every ~100 steps)
            burst_noise = 0.0
            if t % 100 == 0 and np.random.random() < 0.3:
                burst_noise = np.random.uniform(0.2, 0.4)
            
            # Combine all factors
            p_busy = p_base * 0.3 + p_persist * 0.5 + seasonality + burst_noise
            
            # Clip to reasonable bounds
            p_busy = np.clip(p_busy, 0.05, 0.95)
            
            # Generate occupancy
            O[t, c] = 1 if np.random.random() < p_busy else 0
    
    return O


def simulate_demand(T=600, N=12, seed=123):
    """
    Simulate user demand over time.
    
    Args:
        T: Number of time steps
        N: Number of users
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray[float] shape (T, N): Demand matrix in [0, 1]
    """
    np.random.seed(seed)
    
    # Per-user base demand levels
    base_demand = np.random.uniform(0.3, 0.6, N)
    
    # Initialize demand matrix
    D = np.zeros((T, N))
    
    for t in range(T):
        for u in range(N):
            # Base demand with some time variation
            demand = base_demand[u]
            
            # Add some temporal variation
            time_variation = 0.1 * np.sin(2 * np.pi * t / 120 + u)  # Different phase per user
            
            # Occasional demand spikes (every ~50 steps per user)
            spike = 0.0
            if t % 50 == u % 50 and np.random.random() < 0.2:
                spike = np.random.uniform(0.2, 0.4)
            
            # Small random noise
            noise = np.random.normal(0, 0.05)
            
            # Combine and clip to [0, 1]
            D[t, u] = np.clip(demand + time_variation + spike + noise, 0, 1)
    
    return D


def generate_sample_data():
    """Generate and save sample occupancy and demand data for instant demo."""
    print("Generating sample data...")
    
    # Create data directory
    os.makedirs("data/samples", exist_ok=True)
    
    # Generate smaller sample for quick demo
    T_sample, M_sample, N_sample = 200, 6, 10
    
    O_sample = simulate_occupancy(T=T_sample, M=M_sample, seed=42)
    D_sample = simulate_demand(T=T_sample, N=N_sample, seed=123)
    
    # Save to file
    np.savez_compressed(
        "data/samples/sample_run.npz",
        occupancy=O_sample,
        demand=D_sample,
        T=T_sample,
        M=M_sample,
        N=N_sample
    )
    
    print(f"Saved sample data: {T_sample} steps, {M_sample} channels, {N_sample} users")
    print(f"Average occupancy: {O_sample.mean():.2f}")
    print(f"Average demand: {D_sample.mean():.2f}")


if __name__ == "__main__":
    # Generate sample data when run directly
    generate_sample_data()
    
    # Quick test
    print("\nTesting simulation functions...")
    O = simulate_occupancy(T=100, M=4, seed=42)
    D = simulate_demand(T=100, N=6, seed=123)
    
    print(f"Occupancy shape: {O.shape}, range: [{O.min()}, {O.max()}]")
    print(f"Demand shape: {D.shape}, range: [{D.min():.3f}, {D.max():.3f}]")
