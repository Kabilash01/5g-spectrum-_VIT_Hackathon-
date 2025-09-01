"""
Spectrum Optimizer - Interactive Streamlit Dashboard
Real-time simulation and visualization of ML-driven spectrum allocation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# Import our modules
from src.simulate import simulate_occupancy, simulate_demand
from src.features import build_channel_dataset
from src.forecast import train_per_channel_models, predict_p_free
from src.run_loop import run_sim, calculate_performance_lift
from src.metrics import print_method_summary


def load_sample_data():
    """Load pre-generated sample data if available."""
    sample_path = "data/samples/sample_run.npz"
    if os.path.exists(sample_path):
        data = np.load(sample_path)
        return data['occupancy'], data['demand'], int(data['T']), int(data['M']), int(data['N'])
    return None, None, None, None, None


def create_occupancy_heatmap(O, title="Spectrum Occupancy"):
    """Create matplotlib heatmap of spectrum occupancy."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create heatmap (transpose for proper orientation)
    im = ax.imshow(O.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    
    # Customize plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Channel', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Occupancy (0=Free, 1=Busy)', fontsize=11)
    
    # Set y-axis labels to show channel numbers
    ax.set_yticks(range(O.shape[1]))
    ax.set_yticklabels([f'Ch {i}' for i in range(O.shape[1])])
    
    plt.tight_layout()
    return fig


def create_throughput_plot(per_step_data):
    """Create line plot comparing throughput over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    time_steps = per_step_data['time_steps']
    
    # Plot each method
    for method, throughput in per_step_data['throughput'].items():
        ax.plot(time_steps, throughput, label=method, linewidth=2)
    
    ax.set_title('Throughput Comparison Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Throughput (Demand Served)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_user_table(methods):
    """Create DataFrame showing per-user served demand."""
    data = []
    
    for method_name, results in methods.items():
        served = results['served_by_user']
        for user_idx, demand_served in enumerate(served):
            data.append({
                'Method': method_name,
                'User': f'User {user_idx}',
                'Demand Served': round(demand_served, 3)
            })
    
    df = pd.DataFrame(data)
    
    # Pivot for better display
    pivot_df = df.pivot(index='User', columns='Method', values='Demand Served')
    return pivot_df


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Spectrum Optimizer",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("📡 Spectrum Optimizer")
    st.markdown("""
    **AI-powered radio spectrum management simulation**
    
    This demo simulates crowded radio spectrum conditions and compares different channel allocation strategies:
    - **ML Greedy**: Uses machine learning to predict channel availability and allocate smartly
    - **Random**: Baseline random allocation
    - **Round-Robin**: Baseline round-robin allocation
    """)
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    # Load sample data option
    use_sample = st.sidebar.checkbox("Use pre-generated sample data", value=True)
    
    if not use_sample:
        # User-defined parameters
        T = st.sidebar.slider("Time Steps (T)", min_value=200, max_value=1000, value=600, step=50)
        M = st.sidebar.slider("Channels (M)", min_value=4, max_value=16, value=8)
        N = st.sidebar.slider("Users (N)", min_value=4, max_value=30, value=12)
        w = st.sidebar.slider("Lag Window (w)", min_value=5, max_value=30, value=10)
        seed = st.sidebar.number_input("Random Seed", value=42, step=1)
    else:
        # Try to load sample data
        sample_O, sample_D, sample_T, sample_M, sample_N = load_sample_data()
        if sample_O is not None:
            T, M, N = sample_T, sample_M, sample_N
            w = 10
            seed = 42
            st.sidebar.success(f"Sample data loaded: {T} steps, {M} channels, {N} users")
        else:
            st.sidebar.warning("Sample data not found, using default parameters")
            T, M, N, w, seed = 600, 8, 12, 10, 42
    
    # Run simulation button
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary")
    
    if run_button:
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Generate or load data
            status_text.text("Step 1/5: Generating spectrum data...")
            progress_bar.progress(20)
            
            if use_sample and sample_O is not None:
                O, D = sample_O, sample_D
            else:
                O = simulate_occupancy(T=T, M=M, seed=seed)
                D = simulate_demand(T=T, N=N, seed=seed+1)
            
            # Step 2: Build features
            status_text.text("Step 2/5: Engineering features...")
            progress_bar.progress(40)
            
            datasets = build_channel_dataset(O, w=w)
            
            # Step 3: Train models
            status_text.text("Step 3/5: Training ML models...")
            progress_bar.progress(60)
            
            models = train_per_channel_models(datasets)
            
            # Step 4: Predict probabilities
            status_text.text("Step 4/5: Predicting channel availability...")
            progress_bar.progress(80)
            
            p_free = predict_p_free(models, O, w=w)
            
            # Step 5: Run simulation
            status_text.text("Step 5/5: Running allocation simulation...")
            progress_bar.progress(90)
            
            start_time = time.time()
            methods, per_step_data = run_sim(O, D, p_free, seed=seed+2)
            sim_time = time.time() - start_time
            
            # Calculate performance metrics
            lift_stats = calculate_performance_lift(methods)
            
            progress_bar.progress(100)
            status_text.text("✅ Simulation complete!")
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.success(f"Simulation completed in {sim_time:.2f} seconds")
            
            # Key Performance Indicators
            st.header("📊 Key Performance Indicators")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Throughput (ML Greedy)",
                    f"{methods['ML Greedy']['throughput']:.2f}",
                    delta=f"{lift_stats['throughput_lift_pct']:+.1f}% vs baseline"
                )
            
            with col2:
                st.metric(
                    "Collisions (ML Greedy)", 
                    f"{methods['ML Greedy']['collisions']:,}",
                    delta=f"{-lift_stats['collision_reduction_pct']:+.1f}% vs baseline"
                )
            
            with col3:
                st.metric(
                    "Fairness (Jain's Index)",
                    f"{methods['ML Greedy']['fairness']:.3f}",
                    help="1.0 = perfectly fair, lower = less fair"
                )
            
            with col4:
                st.metric(
                    "Success Rate",
                    f"{methods['ML Greedy']['success_rate']:.1%}",
                    help="Percentage of transmission attempts that succeeded"
                )
            
            # Occupancy Heatmaps
            st.header("🌡️ Spectrum Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("True Spectrum Occupancy")
                fig_occupancy = create_occupancy_heatmap(O, "True Spectrum Occupancy")
                st.pyplot(fig_occupancy)
                st.caption("Red = Busy, Blue = Free. Shows actual spectrum usage patterns.")
            
            with col2:
                st.subheader("Predicted Free Probability")
                fig_prediction = create_occupancy_heatmap(p_free, "ML Predicted Free Probability")
                st.pyplot(fig_prediction)
                st.caption("Red = Likely Busy, Blue = Likely Free. ML predictions used for allocation.")
            
            # Throughput Comparison
            st.header("📈 Performance Comparison")
            
            fig_throughput = create_throughput_plot(per_step_data)
            st.pyplot(fig_throughput)
            st.caption("Real-time throughput comparison showing ML advantage over time.")
            
            # Detailed Results Table
            st.header("📋 Detailed Results")
            
            # Method comparison table
            results_data = []
            for method_name, results in methods.items():
                results_data.append({
                    'Method': method_name,
                    'Total Throughput': f"{results['throughput']:.3f}",
                    'Collisions': results['collisions'],
                    'Success Rate': f"{results['success_rate']:.1%}",
                    'Fairness (Jain)': f"{results['fairness']:.3f}"
                })
            
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Per-user served demand
            st.subheader("Per-User Served Demand")
            user_df = create_user_table(methods)
            st.dataframe(user_df, use_container_width=True)
            st.caption("Total demand served to each user by different allocation methods.")
            
            # Technical Details
            with st.expander("🔧 Technical Details"):
                st.write("**Simulation Parameters:**")
                st.write(f"- Time steps: {T}")
                st.write(f"- Channels: {M}") 
                st.write(f"- Users: {N}")
                st.write(f"- Feature window: {w}")
                st.write(f"- Random seed: {seed}")
                st.write(f"- Simulation time: {sim_time:.2f}s")
                
                st.write("**Model Training:**")
                n_trained = sum(1 for model in models if model is not None)
                st.write(f"- Channels with ML models: {n_trained}/{M}")
                st.write(f"- Channels using heuristics: {M - n_trained}/{M}")
                
                st.write("**Performance Lift:**")
                st.write(f"- Best baseline: {lift_stats['best_baseline_name']}")
                st.write(f"- Throughput improvement: {lift_stats['throughput_lift_pct']:+.1f}%")
                st.write(f"- Collision reduction: {lift_stats['collision_reduction_pct']:+.1f}%")
        
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.exception(e)
    
    else:
        # Show welcome message
        st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to start!")
        
        # Show sample information if available
        sample_O, sample_D, sample_T, sample_M, sample_N = load_sample_data()
        if sample_O is not None:
            st.markdown("### 📄 Sample Data Available")
            st.write(f"Pre-generated sample with {sample_T} time steps, {sample_M} channels, and {sample_N} users is ready for instant demo.")


if __name__ == "__main__":
    main()
