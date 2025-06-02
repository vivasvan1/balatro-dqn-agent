#!/usr/bin/env python3
"""
Setup script for MLflow integration with Balatro DQN Agent
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages"""
    print("📦 Installing MLflow and dependencies...")

    requirements = [
        "mlflow>=2.0.0",
        "torch>=1.9.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "pydantic>=1.8.0",
        "scikit-learn>=1.0.0",
    ]

    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")
            return False

    return True


def setup_mlflow_directory():
    """Create MLflow tracking directory"""
    tracking_dir = "mlflow_tracking"
    if not os.path.exists(tracking_dir):
        os.makedirs(tracking_dir)
        print(f"📁 Created MLflow tracking directory: {tracking_dir}")
    else:
        print(f"📁 MLflow tracking directory already exists: {tracking_dir}")


def start_mlflow_ui():
    """Start MLflow UI"""
    print("\n🚀 Starting MLflow UI...")
    print("📊 MLflow UI will be available at: http://localhost:6000")
    print("🔍 You can view your experiments, models, and metrics there")
    print("\n⚠️  Keep this terminal open to maintain the MLflow UI")
    print("💡 Press Ctrl+C to stop the MLflow UI\n")

    try:
        # Set tracking URI to local directory
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = f"file://{os.path.abspath('mlflow_tracking')}"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--host",
                "localhost",
                "--port",
                "6000",
            ],
            env=env,
        )
    except KeyboardInterrupt:
        print("\n🛑 MLflow UI stopped")
    except Exception as e:
        print(f"❌ Error starting MLflow UI: {e}")


def main():
    """Main setup function"""
    print("🎯 Setting up MLflow for Balatro DQN Agent")
    print("=" * 50)

    # Install requirements
    # if not install_requirements():
    #     print("❌ Failed to install requirements. Exiting.")
    #     return

    # Setup directories
    setup_mlflow_directory()

    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("1. Start your Balatro DQN agent API: python src/balatro_agent/main.py")
    print("2. Start MLflow UI in a separate terminal: python setup_mlflow.py --ui-only")
    print("3. Begin training and view metrics at http://localhost:6000")

    # Ask if user wants to start UI now
    start_ui = input("\n🤔 Start MLflow UI now? (y/n): ").lower().strip()
    if start_ui in ["y", "yes"]:
        start_mlflow_ui()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ui-only":
        setup_mlflow_directory()
        start_mlflow_ui()
    else:
        main()
