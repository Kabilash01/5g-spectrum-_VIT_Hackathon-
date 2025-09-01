# 📡 Spectrum Optimizer

**AI-powered radio spectrum management without hardware**

A complete, runnable simulation that demonstrates how machine learning can optimize crowded radio spectrum allocation to maximize throughput and reduce collisions.

## Problem Overview

**Layman's explanation:** Imagine a busy Wi-Fi environment with many devices competing for limited channels. This demo shows how AI can predict which channels will be free and smartly assign devices to reduce interference.

**Technical summary:** ML-based spectrum allocation using per-channel LogisticRegression models to predict next-step occupancy and greedy allocation algorithm for throughput optimization.

## 🚀 No Hardware Required

This is a **pure software simulation** that generates realistic spectrum occupancy patterns, user demands, and interference scenarios without any radio hardware. Everything runs locally on your machine.

### Synthetic Simulator Features:
- **Realistic spectrum patterns**: Markov persistence, seasonality, burst noise
- **Dynamic user demands**: Base demand levels with spikes and temporal variation  
- **Multiple allocation strategies**: ML Greedy vs Random vs Round-Robin baselines
- **Comprehensive metrics**: Throughput, collisions, fairness (Jain's index)

## 📦 Installation

### Windows (PowerShell)
```powershell
# Clone or download this repository
cd spectrum-optimizer

# Create virtual environment
python -m venv .venv

# Activate virtual environment  
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### macOS/Linux
```bash
# Clone or download this repository
cd spectrum-optimizer

# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

```bash
# Run the Streamlit dashboard
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Click **"Run Simulation"** to see the AI in action!

## 📊 What You'll See

### Interactive Dashboard Features:

1. **Spectrum Visualization**: 
   - **True Occupancy Heatmap**: Actual busy/free patterns over time
   - **Predicted Free Probability**: ML model predictions used for allocation

2. **Performance Metrics**:
   - **Total Throughput**: Demand successfully served
   - **Collision Count**: Failed transmissions due to interference
   - **Fairness Index**: How equally users are served (Jain's index)
   - **Success Rate**: Percentage of successful transmissions

3. **Real-time Comparison**:
   - **Line plot**: Throughput over time for all three methods
   - **Performance lift**: How much better ML is than baselines
   - **Per-user analysis**: Demand served to each user

## 🔬 How It Works

### Metrics Definitions

- **Throughput**: Total user demand successfully served (higher = better)
- **Collisions**: Number of users whose transmissions failed due to channel conflicts or occupancy
- **Jain's Fairness Index**: Measures how equally users are served
  - 1.0 = perfectly fair (all users served equally)
  - 1/N = perfectly unfair (only one user served)

### Allocation Methods

1. **ML Greedy** (Our approach):
   - Trains LogisticRegression per channel to predict P(free at next step)
   - Sorts users by demand (highest first)
   - Assigns each user to untaken channel maximizing `demand × P(free)`

2. **Random Baseline**:
   - Randomly shuffles users and channels
   - Assigns first-come-first-served

3. **Round-Robin Baseline**:
   - Cycles through channels in order
   - Assigns high-demand users first

### Why ML Beats Baselines

- **Pattern Recognition**: Learns each channel's behavior (busy periods, persistence, seasonality)
- **Smart Allocation**: Considers both user demand AND channel availability probability
- **Collision Avoidance**: Predicts and avoids likely-busy channels
- **Fairness**: Greedy-by-demand approach naturally serves high-priority users while maintaining fairness

## 🎛️ Customization

Use the sidebar controls to experiment:

- **Time Steps (T)**: Simulation length (200-1000)
- **Channels (M)**: Number of radio channels (4-16) 
- **Users (N)**: Number of competing users (4-30)
- **Lag Window (w)**: Historical window for ML features (5-30)
- **Random Seed**: For reproducible results

## 🚀 Stretch Ideas

Future enhancements could include:

- **Hungarian Algorithm**: Optimal assignment (higher complexity)
- **Contextual Bandit**: Online learning with exploration/exploitation
- **Deep Q-Network (DQN)**: Reinforcement learning approach
- **Multi-band spectrum**: Extend beyond single band
- **Mobility models**: Users moving between coverage areas
- **Real spectrum data**: Integration with actual measurements

## 🛠️ Technical Architecture

```
spectrum-optimizer/
├── app.py                    # Streamlit dashboard (entry point)
├── src/
│   ├── simulate.py          # Spectrum occupancy & demand simulation
│   ├── features.py          # ML feature engineering (lags, rolling stats)
│   ├── forecast.py          # Per-channel LogisticRegression training
│   ├── allocate.py          # Allocation algorithms (greedy, random, RR)
│   ├── metrics.py           # Performance evaluation (throughput, fairness)
│   └── run_loop.py          # Main simulation orchestration
├── data/samples/            # Pre-generated demo data
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 📈 Expected Results

On typical runs, you should see:
- **ML Greedy throughput**: 10-30% higher than best baseline
- **Collision reduction**: 15-40% fewer collisions than baselines  
- **Sub-second simulation**: Fast enough for interactive experimentation
- **Robust performance**: Works even with degenerate channels (always busy/free)

## 📄 License

This project is open source. Feel free to experiment, modify, and extend!

---

**🎯 Ready to optimize some spectrum?** Run `streamlit run app.py` and click that big green button!
