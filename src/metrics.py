"""
Performance metrics for spectrum allocation evaluation.
Calculates throughput, collisions, fairness, and other KPIs.
"""

import numpy as np


def step_throughput(assignment, D_t, O_next):
    """
    Calculate throughput, collisions, and attempts for one time step.
    
    A user succeeds if:
    1. Assigned to a channel (assignment[user] >= 0)
    2. Channel is free at next time step (O_next[channel] == 0)
    3. No other user assigned to same channel (no collision)
    
    Args:
        assignment: Channel assignments, shape (N,) with values in {-1, 0, 1, ..., M-1}
        D_t: User demands at time t, shape (N,)
        O_next: True occupancy at next time step, shape (M,)
    
    Returns:
        Tuple of (throughput, collisions, attempts):
        - throughput: Total successful demand served
        - collisions: Number of users that failed due to collisions
        - attempts: Number of users that attempted transmission
    """
    N = len(D_t)
    M = len(O_next)
    
    throughput = 0.0
    collisions = 0
    attempts = 0
    
    # Group users by assigned channel
    channel_users = {}
    for user in range(N):
        channel = assignment[user]
        if channel >= 0:  # User is assigned to a channel
            if channel not in channel_users:
                channel_users[channel] = []
            channel_users[channel].append(user)
            attempts += 1
    
    # Process each channel
    for channel, users in channel_users.items():
        if channel >= M:  # Invalid channel assignment
            collisions += len(users)
            continue
            
        channel_free = (O_next[channel] == 0)
        multiple_users = len(users) > 1
        
        if channel_free and not multiple_users:
            # Success: channel free and only one user
            user = users[0]
            throughput += D_t[user]
        else:
            # Failure: either channel busy or collision
            collisions += len(users)
    
    return throughput, collisions, attempts


def jain_index(served_by_user):
    """
    Calculate Jain's fairness index.
    
    Jain's index = (sum(x_i))^2 / (n * sum(x_i^2))
    where x_i is the total demand served for user i.
    
    Range: [1/n, 1] where n is number of users
    - 1 = perfectly fair (all users served equally)
    - 1/n = perfectly unfair (only one user served)
    
    Args:
        served_by_user: Total demand served per user, shape (N,)
    
    Returns:
        float: Jain's fairness index
    """
    if len(served_by_user) == 0:
        return 1.0
    
    # Handle case where all users have zero service
    if np.sum(served_by_user) == 0:
        return 1.0  # Technically fair (everyone gets nothing)
    
    n = len(served_by_user)
    sum_x = np.sum(served_by_user)
    sum_x_squared = np.sum(served_by_user ** 2)
    
    if sum_x_squared == 0:
        return 1.0
    
    fairness = (sum_x ** 2) / (n * sum_x_squared)
    return fairness


def accumulate_metrics(method_results, assignment, D_t, O_next):
    """
    Update accumulated metrics with results from one time step.
    
    Args:
        method_results: Dict to accumulate results in
        assignment: Channel assignments for this step
        D_t: User demands for this step
        O_next: True occupancy at next step
    
    Returns:
        Tuple of (step_throughput, step_collisions, step_attempts)
    """
    step_tp, step_coll, step_att = step_throughput(assignment, D_t, O_next)
    
    # Accumulate totals
    method_results['throughput'] += step_tp
    method_results['collisions'] += step_coll
    method_results['attempts'] += step_att
    
    # Update per-user served demand
    N = len(D_t)
    for user in range(N):
        channel = assignment[user]
        if channel >= 0 and channel < len(O_next):
            # Check if this user succeeded
            channel_users = [u for u in range(N) if assignment[u] == channel]
            channel_free = (O_next[channel] == 0)
            no_collision = len(channel_users) == 1
            
            if channel_free and no_collision:
                method_results['served_by_user'][user] += D_t[user]
    
    return step_tp, step_coll, step_att


def init_method_results(N):
    """
    Initialize results dictionary for one allocation method.
    
    Args:
        N: Number of users
    
    Returns:
        Dict with initialized metrics
    """
    return {
        'throughput': 0.0,
        'collisions': 0,
        'attempts': 0,
        'served_by_user': np.zeros(N),
        'per_step_throughput': []  # For plotting
    }


def finalize_method_results(method_results):
    """
    Calculate final metrics after simulation completes.
    
    Args:
        method_results: Results dict from accumulate_metrics
    
    Returns:
        Dict with final computed metrics
    """
    # Calculate fairness
    fairness = jain_index(method_results['served_by_user'])
    method_results['fairness'] = fairness
    
    # Calculate collision rate
    if method_results['attempts'] > 0:
        collision_rate = method_results['collisions'] / method_results['attempts']
    else:
        collision_rate = 0.0
    method_results['collision_rate'] = collision_rate
    
    # Calculate success rate  
    success_rate = 1.0 - collision_rate
    method_results['success_rate'] = success_rate
    
    return method_results


def print_method_summary(method_name, results):
    """
    Print summary of results for one allocation method.
    
    Args:
        method_name: Name of the allocation method
        results: Results dict from finalize_method_results
    """
    print(f"\n{method_name} Results:")
    print(f"  Total Throughput: {results['throughput']:.3f}")
    print(f"  Total Collisions: {results['collisions']}")
    print(f"  Total Attempts: {results['attempts']}")
    print(f"  Success Rate: {results['success_rate']:.1%}")
    print(f"  Collision Rate: {results['collision_rate']:.1%}")
    print(f"  Fairness (Jain): {results['fairness']:.3f}")
    print(f"  Per-user served: {results['served_by_user']}")


if __name__ == "__main__":
    print("Testing metrics calculation...")
    
    # Test scenario
    N, M = 4, 3
    
    # Sample data for one time step
    assignment = np.array([0, 1, 0, -1])  # Users 0,2 -> channel 0, user 1 -> channel 1, user 3 unassigned
    D_t = np.array([0.8, 0.6, 0.4, 0.9])
    O_next = np.array([0, 0, 1])  # Channels 0,1 free, channel 2 busy
    
    print(f"\nTest scenario:")
    print(f"Assignment: {assignment}")
    print(f"Demands: {D_t}")
    print(f"Next occupancy: {O_next}")
    
    # Calculate step metrics
    tp, coll, att = step_throughput(assignment, D_t, O_next)
    print(f"\nStep results:")
    print(f"Throughput: {tp:.3f}")
    print(f"Collisions: {coll}")
    print(f"Attempts: {att}")
    
    # Test fairness calculation
    served_equal = np.array([1.0, 1.0, 1.0, 1.0])
    served_unequal = np.array([4.0, 0.0, 0.0, 0.0])
    served_mixed = np.array([2.0, 1.5, 1.0, 0.5])
    
    print(f"\nFairness tests:")
    print(f"Equal service {served_equal}: {jain_index(served_equal):.3f}")
    print(f"Unequal service {served_unequal}: {jain_index(served_unequal):.3f}")
    print(f"Mixed service {served_mixed}: {jain_index(served_mixed):.3f}")
    
    # Test accumulation
    results = init_method_results(N)
    accumulate_metrics(results, assignment, D_t, O_next)
    results = finalize_method_results(results)
    print_method_summary("Test Method", results)
