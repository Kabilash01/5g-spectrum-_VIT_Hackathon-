"""
Main simulation loop orchestrating spectrum allocation comparison.
Runs allocation algorithms and accumulates metrics over time.
"""

import numpy as np
from .allocate import greedy_allocator, random_allocator, round_robin_allocator
from .metrics import init_method_results, accumulate_metrics, finalize_method_results


def run_sim(O, D, p_free, seed=7):
    """
    Run complete simulation comparing allocation methods.
    
    Args:
        O: True occupancy matrix, shape (T, M)
        D: User demand matrix, shape (T, N)  
        p_free: Predicted free probabilities, shape (T, M)
        seed: Random seed for reproducible comparisons
    
    Returns:
        Tuple of (method_summaries, per_step_data):
        - method_summaries: Dict with results for each method
        - per_step_data: Dict with per-step throughput arrays for plotting
    """
    T, M = O.shape
    T_demand, N = D.shape
    
    assert T == T_demand, f"Occupancy and demand time dimensions don't match: {T} vs {T_demand}"
    
    # Initialize random number generator
    rng = np.random.RandomState(seed)
    
    # Initialize results for each method
    methods = {
        'ML Greedy': init_method_results(N),
        'Random': init_method_results(N), 
        'Round-Robin': init_method_results(N)
    }
    
    # Round-robin state
    rr_ptr = 0
    
    # Skip first few time steps (not enough history for features)
    w = 10  # Should match the window size used in feature engineering
    
    print(f"Running simulation: T={T}, M={M}, N={N}, starting from t={w}")
    
    # Main simulation loop
    for t in range(w, T-1):  # T-1 because we need O[t+1]
        # Current state
        D_t = D[t, :]           # User demands at time t
        p_t = p_free[t, :]      # Predicted free probabilities at time t
        O_next = O[t+1, :]      # True occupancy at time t+1
        
        # Allocation methods
        
        # 1. ML Greedy: Use predictions to make smart assignments
        assignment_greedy = greedy_allocator(D_t, p_t)
        
        # 2. Random: Ignore predictions, random assignment
        assignment_random = random_allocator(D_t, p_t, rng)
        
        # 3. Round-Robin: Ignore predictions, cycle through channels
        assignment_rr, rr_ptr = round_robin_allocator(D_t, p_t, rr_ptr)
        
        # Evaluate each method
        step_tp_greedy, _, _ = accumulate_metrics(methods['ML Greedy'], assignment_greedy, D_t, O_next)
        step_tp_random, _, _ = accumulate_metrics(methods['Random'], assignment_random, D_t, O_next)
        step_tp_rr, _, _ = accumulate_metrics(methods['Round-Robin'], assignment_rr, D_t, O_next)
        
        # Store per-step throughput for plotting
        methods['ML Greedy']['per_step_throughput'].append(step_tp_greedy)
        methods['Random']['per_step_throughput'].append(step_tp_random)
        methods['Round-Robin']['per_step_throughput'].append(step_tp_rr)
    
    # Finalize results
    for method_name in methods:
        methods[method_name] = finalize_method_results(methods[method_name])
    
    # Prepare per-step data for plotting
    per_step_data = {
        'time_steps': list(range(w, T-1)),
        'throughput': {
            'ML Greedy': methods['ML Greedy']['per_step_throughput'],
            'Random': methods['Random']['per_step_throughput'], 
            'Round-Robin': methods['Round-Robin']['per_step_throughput']
        }
    }
    
    print(f"Simulation complete: {T-1-w} time steps evaluated")
    
    return methods, per_step_data


def calculate_performance_lift(methods):
    """
    Calculate performance lift of ML Greedy vs best baseline.
    
    Args:
        methods: Results dict from run_sim
        
    Returns:
        Dict with lift percentages
    """
    greedy_throughput = methods['ML Greedy']['throughput']
    random_throughput = methods['Random']['throughput']
    rr_throughput = methods['Round-Robin']['throughput']
    
    # Best baseline is max of Random and Round-Robin
    best_baseline = max(random_throughput, rr_throughput)
    
    if best_baseline > 0:
        throughput_lift = ((greedy_throughput - best_baseline) / best_baseline) * 100
    else:
        throughput_lift = 0.0
    
    # Also calculate collision reduction
    greedy_collisions = methods['ML Greedy']['collisions']
    best_baseline_collisions = min(methods['Random']['collisions'], methods['Round-Robin']['collisions'])
    
    if best_baseline_collisions > 0:
        collision_reduction = ((best_baseline_collisions - greedy_collisions) / best_baseline_collisions) * 100
    else:
        collision_reduction = 0.0
    
    return {
        'throughput_lift_pct': throughput_lift,
        'collision_reduction_pct': collision_reduction,
        'best_baseline_name': 'Random' if random_throughput > rr_throughput else 'Round-Robin',
        'best_baseline_throughput': best_baseline
    }


def print_simulation_summary(methods, lift_stats):
    """
    Print comprehensive simulation results.
    
    Args:
        methods: Results from run_sim
        lift_stats: Results from calculate_performance_lift
    """
    print(f"\n{'='*60}")
    print(f"SIMULATION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Print each method's results
    for method_name, results in methods.items():
        print(f"\n{method_name}:")
        print(f"  Throughput: {results['throughput']:.3f}")
        print(f"  Collisions: {results['collisions']}")
        print(f"  Success Rate: {results['success_rate']:.1%}")
        print(f"  Fairness: {results['fairness']:.3f}")
    
    # Print performance lift
    print(f"\n{'='*30}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*30}")
    print(f"Best baseline: {lift_stats['best_baseline_name']} ({lift_stats['best_baseline_throughput']:.3f})")
    print(f"ML Greedy throughput: {methods['ML Greedy']['throughput']:.3f}")
    print(f"Throughput lift: {lift_stats['throughput_lift_pct']:+.1f}%")
    print(f"Collision reduction: {lift_stats['collision_reduction_pct']:+.1f}%")


if __name__ == "__main__":
    # Test with simulated data
    from .simulate import simulate_occupancy, simulate_demand
    from .features import build_channel_dataset
    from .forecast import train_per_channel_models, predict_p_free
    
    print("Testing complete simulation loop...")
    
    # Generate test data
    T, M, N = 100, 4, 6
    O = simulate_occupancy(T=T, M=M, seed=42)
    D = simulate_demand(T=T, N=N, seed=123)
    
    # Train models and predict
    w = 10
    datasets = build_channel_dataset(O, w=w)
    models = train_per_channel_models(datasets)
    p_free = predict_p_free(models, O, w=w)
    
    # Run simulation
    methods, per_step_data = run_sim(O, D, p_free, seed=7)
    
    # Calculate and print results
    lift_stats = calculate_performance_lift(methods)
    print_simulation_summary(methods, lift_stats)
    
    # Quick check of per-step data
    print(f"\nPer-step data shape: {len(per_step_data['time_steps'])} time steps")
    for method in per_step_data['throughput']:
        avg_step_tp = np.mean(per_step_data['throughput'][method])
        print(f"{method} avg step throughput: {avg_step_tp:.3f}")
