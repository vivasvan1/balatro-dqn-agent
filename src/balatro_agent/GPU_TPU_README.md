# GPU/TPU Optimized Balatro DQN Training

This training script has been optimized for GPU and TPU acceleration to significantly speed up training times and improve performance.

## 🚀 Key Optimizations

### 1. **Mixed Precision Training**
- Uses `torch.cuda.amp` for automatic mixed precision (FP16/FP32)
- Reduces memory usage by ~50% and speeds up training by 1.5-2x
- Automatically enabled when GPU/TPU is available

### 2. **Larger Network Architecture**
- Increased layer sizes (256 → 256 → 128) for better GPU utilization
- Added Batch Normalization for faster convergence
- Kaiming weight initialization for better training stability

### 3. **Optimized Hyperparameters**
- **Larger batch size**: 128 (vs 64 for CPU) for better GPU utilization
- **Larger replay buffer**: 50,000 (vs 10,000) for more diverse experiences
- **Gradient accumulation**: 2 steps for effective larger batch sizes
- **AdamW optimizer**: Better weight decay and convergence

### 4. **Memory and Performance Optimizations**
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling with ReduceLROnPlateau
- Efficient tensor operations with `torch.no_grad()` for inference
- Memory-efficient experience replay

### 5. **Hardware Detection**
- Automatic detection of CUDA GPUs
- TPU support (requires `torch_xla`)
- Fallback to CPU if no acceleration available

## 🎮 GPU Requirements

### NVIDIA GPU
- **CUDA 11.8+** compatible GPU
- **8GB+ VRAM** recommended (4GB minimum)
- **Latest NVIDIA drivers**

### TPU (Google Cloud)
- **Google Cloud TPU v2/v3/v4**
- **torch_xla** package installed
- **Cloud TPU runtime** configured

## 📦 Installation

### Quick Setup
```bash
# Run the automated setup script
python setup_gpu.py
```

### Manual Installation

#### For NVIDIA GPU:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements_gpu.txt
```

#### For TPU:
```bash
# Install TPU dependencies
pip install torch_xla
pip install torch_xla[tpu]

# Install other dependencies
pip install -r requirements_gpu.txt
```

#### For CPU only:
```bash
# Install PyTorch for CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements_gpu.txt
```

## 🚀 Usage

### Start Training
```bash
python train_balatro_v2_simple.py
```

The script will automatically:
1. Detect available hardware (GPU/TPU/CPU)
2. Configure optimizations accordingly
3. Display hardware information and settings
4. Start training with appropriate batch sizes and memory settings

### Expected Performance Improvements

| Hardware | Episodes/Second | Memory Usage | Training Time (100k episodes) |
|----------|----------------|--------------|------------------------------|
| CPU      | ~5-10          | ~2GB RAM     | ~3-6 hours                   |
| GPU      | ~20-50         | ~4-8GB VRAM  | ~30-60 minutes               |
| TPU      | ~50-100        | ~8-16GB      | ~15-30 minutes               |

*Performance varies based on hardware specifications*

## 📊 Monitoring

### Real-time Metrics
The training script displays:
- **Episodes per second** (speed metric)
- **GPU memory usage** (if applicable)
- **Training progress** with running averages
- **Hardware utilization** statistics

### MLflow Integration
- Automatic logging of all metrics
- Model checkpointing
- Performance plots and diagnostics
- Artifact storage for best models

## 🔧 Configuration

### Environment Variables
```bash
# Force CPU usage (if you want to disable GPU)
export CUDA_VISIBLE_DEVICES=""

# Set specific GPU
export CUDA_VISIBLE_DEVICES="0"

# TPU configuration
export TPU_NAME="your-tpu-name"
```

### Script Parameters
Key parameters in `train_balatro_v2_simple.py`:

```python
# GPU/TPU specific settings
BATCH_SIZE = 128 if GPU_AVAILABLE or TPU_AVAILABLE else 64
MEMORY_SIZE = 50000 if GPU_AVAILABLE or TPU_AVAILABLE else 10000
MIXED_PRECISION = True
GRADIENT_ACCUMULATION_STEPS = 2 if GPU_AVAILABLE or TPU_AVAILABLE else 1
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size
BATCH_SIZE = 64  # Instead of 128

# Reduce memory buffer
MEMORY_SIZE = 25000  # Instead of 50000
```

#### TPU Connection Issues
```bash
# Check TPU status
gcloud compute tpus list

# Restart TPU if needed
gcloud compute tpus stop your-tpu-name
gcloud compute tpus start your-tpu-name
```

#### Performance Issues
1. **Check GPU utilization**: `nvidia-smi`
2. **Monitor memory usage**: Watch for memory leaks
3. **Verify mixed precision**: Check if `scaler` is not None
4. **Profile training**: Use PyTorch profiler

### Debug Mode
Add debug prints to monitor performance:
```python
# Add to training loop
if episode % 100 == 0:
    if GPU_AVAILABLE:
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    print(f"Speed: {episodes_per_sec:.1f} ep/s")
```

## 📈 Performance Tips

### For Maximum Speed
1. **Use latest PyTorch version** (2.0+)
2. **Enable mixed precision** (automatic)
3. **Use larger batch sizes** (if memory allows)
4. **Monitor GPU utilization** (should be >80%)
5. **Use SSD storage** for faster data loading

### For Memory Efficiency
1. **Reduce batch size** if OOM occurs
2. **Use gradient accumulation** for effective larger batches
3. **Monitor memory usage** with `nvidia-smi`
4. **Clear cache** periodically: `torch.cuda.empty_cache()`

## 🎯 Best Practices

1. **Start with default settings** - they're optimized for most hardware
2. **Monitor training speed** - aim for >20 episodes/second on GPU
3. **Check convergence** - use the built-in diagnostics
4. **Save checkpoints** - MLflow handles this automatically
5. **Use appropriate hardware** - GPU recommended for serious training

## 📚 Additional Resources

- [PyTorch GPU Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [TPU PyTorch Guide](https://pytorch.org/xla/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

**Happy Training! 🎰⚡** 