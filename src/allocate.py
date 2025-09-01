"""
Channel allocation algorithms.
Implements greedy, random, and round-robin strategies for assigning users to channels.
"""

import numpy as np


def greedy_allocator(D_t, p_free_t):
    """
    Greedy allocation: assign users to channels to maximize expected throughput.
    
    Strategy:
    1. Sort users by demand (highest first)
    2. For each user, assign to the untaken channel that maximizes demand * p_free
    3. Each channel can only have one user
    
    Args:
        D_t: User demands at time t, shape (N,)
        p_free_t: Channel free probabilities at time t, shape (M,)
    
    Returns:
        np.ndarray shape (N,): Channel assignment per user (-1 if unassigned)
    """
    N = len(D_t)
    M = len(p_free_t)
    
    # Initialize assignments
    assignment = np.full(N, -1, dtype=int)
    taken_channels = set()
    
    # Sort users by demand (descending)
    user_order = np.argsort(-D_t)  # Negative for descending order
    
    for user in user_order:
        best_channel = -1
        best_score = -1
        
        # Find best available channel for this user
        for channel in range(M):
            if channel not in taken_channels:
                score = D_t[user] * p_free_t[channel]
                if score > best_score:
                    best_score = score
                    best_channel = channel
        
        # Assign user to best channel (if any available)
        if best_channel >= 0:
            assignment[user] = best_channel
            taken_channels.add(best_channel)
    
    return assignment


def random_allocator(D_t, p_free_t, rng):
    """
    Random allocation: randomly assign users to available channels.
    
    Args:
        D_t: User demands at time t, shape (N,)
        p_free_t: Channel free probabilities at time t, shape (M,) [unused]
        rng: Random number generator (np.random.RandomState)
    
    Returns:
        np.ndarray shape (N,): Channel assignment per user (-1 if unassigned)
    """
    N = len(D_t)
    M = len(p_free_t)
    
    # Initialize assignments
    assignment = np.full(N, -1, dtype=int)
    
    # Create list of available channels
    available_channels = list(range(M))
    rng.shuffle(available_channels)
    
    # Randomly order users
    user_order = list(range(N))
    rng.shuffle(user_order)
    
    # Assign users to channels (first come, first served)
    channel_idx = 0
    for user in user_order:
        if channel_idx < len(available_channels):
            assignment[user] = available_channels[channel_idx]
            channel_idx += 1
    
    return assignment


def round_robin_allocator(D_t, p_free_t, rr_ptr):
    """
    Round-robin allocation: cycle through channels in order.
    
    Args:
        D_t: User demands at time t, shape (N,)
        p_free_t: Channel free probabilities at time t, shape (M,) [unused]
        rr_ptr: Current round-robin pointer (channel index)
    
    Returns:
        Tuple of (assignment, next_rr_ptr):
        - assignment: np.ndarray shape (N,): Channel assignment per user (-1 if unassigned)
        - next_rr_ptr: Updated round-robin pointer for next time step
    """
    N = len(D_t)
    M = len(p_free_t)
    
    # Initialize assignments
    assignment = np.full(N, -1, dtype=int)
    
    # Sort users by demand (highest first) for fair high-demand user treatment
    user_order = np.argsort(-D_t)
    
    current_ptr = rr_ptr
    
    # Assign users to channels in round-robin fashion
    for i, user in enumerate(user_order):
        if i < M:  # Only assign if we have enough channels
            assignment[user] = current_ptr
            current_ptr = (current_ptr + 1) % M
    
    return assignment, current_ptr


def print_allocation_summary(assignment, D_t, p_free_t, method_name):
    """
    Print summary statistics for an allocation.
    
    Args:
        assignment: Channel assignments, shape (N,)
        D_t: User demands, shape (N,)
        p_free_t: Channel free probabilities, shape (M,)
        method_name: Name of allocation method
    """
    N = len(D_t)
    M = len(p_free_t)
    
    # Count assignments
    n_assigned = np.sum(assignment >= 0)
    n_unassigned = N - n_assigned
    
    # Calculate expected throughput
    expected_throughput = 0.0
    for user in range(N):
        if assignment[user] >= 0:
            channel = assignment[user]
            expected_throughput += D_t[user] * p_free_t[channel]
    
    # Channel utilization
    assigned_channels = set(assignment[assignment >= 0])
    channel_utilization = len(assigned_channels) / M
    
    print(f"\n{method_name} Allocation Summary:")
    print(f"  Assigned users: {n_assigned}/{N}")
    print(f"  Unassigned users: {n_unassigned}")
    print(f"  Channel utilization: {channel_utilization:.1%}")
    print(f"  Expected throughput: {expected_throughput:.3f}")


if __name__ == "__main__":
    print("Testing allocation algorithms...")
    
    # Test scenario
    N, M = 8, 5
    np.random.seed(42)
    
    # Sample demands and channel probabilities
    D_t = np.random.uniform(0.2, 0.8, N)
    p_free_t = np.random.uniform(0.3, 0.9, M)
    
    print(f"\nTest scenario: {N} users, {M} channels")
    print(f"User demands: {D_t}")
    print(f"Channel free probabilities: {p_free_t}")
    
    # Test greedy allocation
    assignment_greedy = greedy_allocator(D_t, p_free_t)
    print_allocation_summary(assignment_greedy, D_t, p_free_t, "Greedy")
    
    # Test random allocation
    rng = np.random.RandomState(42)
    assignment_random = random_allocator(D_t, p_free_t, rng)
    print_allocation_summary(assignment_random, D_t, p_free_t, "Random")
    
    # Test round-robin allocation
    assignment_rr, next_ptr = round_robin_allocator(D_t, p_free_t, rr_ptr=0)
    print_allocation_summary(assignment_rr, D_t, p_free_t, "Round-Robin")
    print(f"Next RR pointer: {next_ptr}")
    
    print(f"\nAssignment comparison:")
    print(f"Greedy:     {assignment_greedy}")
    print(f"Random:     {assignment_random}")
    print(f"Round-Robin: {assignment_rr}")
