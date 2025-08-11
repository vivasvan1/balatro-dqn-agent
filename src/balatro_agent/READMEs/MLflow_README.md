# 🚀 MLflow Integration for Balatro DQN Agent

The custom `PerformanceTracker` has been replaced with **MLflow**, a professional ML experiment tracking platform that provides superior monitoring, visualization, and model management.

## 🌟 Benefits of MLflow vs Custom Tracker

| Feature | Custom Tracker | MLflow |
|---------|---------------|--------|
| **Web UI** | ❌ Basic plots only | ✅ Professional web interface |
| **Experiment Tracking** | ❌ Limited | ✅ Full experiment lifecycle |
| **Model Versioning** | ❌ Manual file naming | ✅ Automatic versioning & registry |
| **Metrics Visualization** | ❌ Static plots | ✅ Interactive plots & comparisons |
| **Scalability** | ❌ Local files only | ✅ Local, remote, cloud support |
| **Industry Standard** | ❌ Custom solution | ✅ Used by major companies |

## 🛠️ Setup Instructions

### 1. Install Dependencies
```bash
# Option A: Run setup script (recommended)
python setup_mlflow.py

# Option B: Manual installation
pip install -r src/balatro_agent/requirements.txt
```

### 2. Start MLflow UI
```bash
# In a separate terminal
python setup_mlflow.py --ui-only

# Or manually
mlflow ui --host localhost --port 5000
```

### 3. Start Your Agent
```bash
# Start the Balatro DQN agent API
python src/balatro_agent/main.py
```

### 4. View Experiments
Open your browser to: **http://localhost:5000**

## 📊 What You'll See in MLflow

### **Experiments Dashboard**
- List of all training runs
- Performance metrics comparison
- Run status and duration

### **Run Details**
- **Metrics**: Episode rewards, training loss, Q-values, epsilon decay
- **Parameters**: All hyperparameters (learning rate, batch size, etc.)
- **Artifacts**: Model weights, performance plots
- **Models**: Versioned model registry

### **Metrics Visualization**
- Interactive plots for all metrics
- Real-time updates during training
- Compare multiple runs side-by-side

## 🎯 Key Features

### **Automatic Model Saving**
- **Best Episode Model**: Saved when new best episode reward achieved
- **Best Average Model**: Saved when new best 10-episode average achieved
- **Model Registry**: Organized model versions with performance metadata

### **Real-time Tracking**
- Episode rewards and lengths
- Training loss over time
- Epsilon decay progression
- Q-value evolution
- Buffer size growth

### **API Endpoints**
- `GET /metrics` - Current performance metrics
- `POST /mlflow/start_run` - Start new experiment run
- `POST /mlflow/end_run` - End current run
- `GET /mlflow/ui_info` - MLflow UI information

## 🔧 Migration from Custom Tracker

The migration is **automatic** - no changes needed to your training code:

```python
# Before (custom tracker)
tracker = PerformanceTracker()
tracker.log_episode_end()

# After (MLflow - same interface!)
tracker = MLflowTracker()
tracker.log_episode_end(agent)  # Just needs agent reference
```

## 📈 Advanced Usage

### **Starting New Experiments**
```python
# Via API
curl -X POST "http://localhost:8000/mlflow/start_run?run_name=experiment_v2"

# Or programmatically
tracker.start_run("hyperparameter_tuning_v1")
```

### **Custom Metrics**
```python
# Log custom metrics
tracker.log_custom_metric("custom_score", 0.85, step=100)

# Log artifacts (files)
tracker.log_artifact("analysis_report.pdf", "reports")
```

### **Model Loading**
```python
import mlflow.pytorch

# Load best model from registry
model = mlflow.pytorch.load_model("models:/balatro_dqn_best_episode/latest")
```

## 🗂️ File Structure

```
├── mlflow_tracking/          # MLflow data storage
│   ├── experiments/          # Experiment metadata
│   └── artifacts/            # Model weights, plots
├── mlflow_tracker.py         # MLflow tracker implementation
├── setup_mlflow.py          # Setup and UI launcher
└── src/balatro_agent/
    ├── main.py              # Updated with MLflow
    └── requirements.txt     # MLflow dependencies
```

## 🆘 Troubleshooting

### **MLflow UI not starting?**
```bash
# Check if port 5000 is available
lsof -i :5000

# Use different port
mlflow ui --port 5001
```

### **Can't see experiments?**
- Ensure `mlflow_tracking` directory exists
- Check tracking URI matches in both agent and UI
- Restart MLflow UI after first run

### **Models not saving?**
- Check MLflow run is active (`tracker.current_run` not None)
- Verify disk space and permissions
- Check agent.qnetwork_local is accessible

## 🎉 Benefits Recap

✅ **Professional UI** - No more basic matplotlib plots  
✅ **Experiment Comparison** - Compare different runs easily  
✅ **Model Registry** - Organized model versioning  
✅ **Real-time Monitoring** - Live updates during training  
✅ **Industry Standard** - Used by Netflix, Microsoft, Databricks  
✅ **Scalable** - Can move to cloud when needed  

**Your training data is now tracked like a pro! 🚀** 