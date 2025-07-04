#!/usr/bin/env python3
"""
Setup script for GPU/TPU optimized Balatro DQN training
Automatically detects hardware and installs appropriate PyTorch version
"""

import subprocess
import sys
import platform
import os

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def check_tpu():
    """Check if TPU is available"""
    try:
        import torch_xla
        return True
    except ImportError:
        return False

def get_system_info():
    """Get system information"""
    system = platform.system()
    machine = platform.machine()
    python_version = sys.version_info
    
    print(f"System: {system}")
    print(f"Architecture: {machine}")
    print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    return system, machine, python_version

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("Installing PyTorch with CUDA support...")
    
    # Get CUDA version
    nvidia_smi = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
    if nvidia_smi:
        print(f"Detected NVIDIA driver version: {nvidia_smi}")
    
    # Install PyTorch with CUDA
    command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    print(f"Running: {command}")
    
    result = run_command(command)
    if result is not None:
        print("✅ PyTorch with CUDA installed successfully!")
        return True
    else:
        print("❌ Failed to install PyTorch with CUDA")
        return False

def install_pytorch_cpu():
    """Install PyTorch for CPU only"""
    print("Installing PyTorch for CPU...")
    
    command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    print(f"Running: {command}")
    
    result = run_command(command)
    if result is not None:
        print("✅ PyTorch for CPU installed successfully!")
        return True
    else:
        print("❌ Failed to install PyTorch for CPU")
        return False

def install_tpu_dependencies():
    """Install TPU dependencies"""
    print("Installing TPU dependencies...")
    
    commands = [
        "pip install torch_xla",
        "pip install torch_xla[tpu]",
    ]
    
    for command in commands:
        print(f"Running: {command}")
        result = run_command(command)
        if result is None:
            print(f"❌ Failed to run: {command}")
            return False
    
    print("✅ TPU dependencies installed successfully!")
    return True

def install_other_dependencies():
    """Install other required dependencies"""
    print("Installing other dependencies...")
    
    command = "pip install -r requirements_gpu.txt"
    print(f"Running: {command}")
    
    result = run_command(command)
    if result is not None:
        print("✅ Other dependencies installed successfully!")
        return True
    else:
        print("❌ Failed to install other dependencies")
        return False

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("ℹ️ CUDA not available")
        
        try:
            import torch_xla
            print("✅ TPU support available")
        except ImportError:
            print("ℹ️ TPU support not installed")
        
        import numpy
        print(f"✅ NumPy version: {numpy.__version__}")
        
        import mlflow
        print(f"✅ MLflow version: {mlflow.__version__}")
        
        print("\n🎉 Installation test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    print("🚀 Balatro DQN GPU/TPU Setup")
    print("=" * 40)
    
    # Get system info
    system, machine, python_version = get_system_info()
    
    # Check current PyTorch installation
    print(f"\n🔍 Checking current installation...")
    cuda_available = check_cuda()
    tpu_available = check_tpu()
    
    if cuda_available:
        print("✅ CUDA support detected")
    if tpu_available:
        print("✅ TPU support detected")
    
    # Ask user what to install
    print(f"\n📋 Installation options:")
    print("1. Install PyTorch with CUDA support (recommended for NVIDIA GPUs)")
    print("2. Install PyTorch for CPU only")
    print("3. Install TPU dependencies (for Google Cloud TPU)")
    print("4. Install all dependencies (PyTorch + other packages)")
    print("5. Test current installation")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        install_pytorch_cuda()
    elif choice == "2":
        install_pytorch_cpu()
    elif choice == "3":
        install_tpu_dependencies()
    elif choice == "4":
        print("Installing all dependencies...")
        if system == "Linux" or system == "Darwin":
            install_pytorch_cuda()
        else:
            install_pytorch_cpu()
        install_other_dependencies()
    elif choice == "5":
        test_installation()
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Test installation
    if choice in ["1", "2", "3", "4"]:
        test_installation()
    
    print(f"\n🎯 Setup complete! You can now run:")
    print(f"   python train_balatro_v2_simple.py")

if __name__ == "__main__":
    main() 