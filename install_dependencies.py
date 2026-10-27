import os
import sys
import struct
import platform
import subprocess

def check_architecture():
    bitness = struct.calcsize("P") * 8
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    if bitness != 64:
        print("=" * 75)
        print("❌ [ERROR] 32-BIT PYTHON DETECTED!")
        print("=" * 75)
        print(f"You are currently running: Python {py_ver} ({platform.architecture()[0]})")
        print("\nTensorFlow DOES NOT support 32-bit Windows Python.")
        print("This is why pip reports:")
        print("   'ERROR: Could not find a version that satisfies the requirement tensorflow'")
        print("   '(from versions: none)'\n")
        print("👉 HOW TO FIX (Takes 2 minutes):")
        print("1. Open Windows Settings -> Apps -> Installed Apps.")
        print("2. Find 'Python' and click 'Uninstall'.")
        print("3. Go to https://www.python.org/downloads/windows/")
        print("4. Download the 'Windows installer (64-bit)' (Python 3.10, 3.11, or 3.12 recommended).")
        print("5. ⚠️ IMPORTANT: In the installer, check '[x] Add python.exe to PATH' at the bottom.")
        print("6. Run 'run.bat' again after installing.")
        print("=" * 75)
        sys.exit(1)
    
    print(f"[OK] Python {py_ver} (64-bit) verified.")

def run_pip(args, desc):
    print(f"\n--> {desc}...")
    cmd = [sys.executable, "-m", "pip"] + args
    result = subprocess.run(cmd)
    return result.returncode == 0

def install_all():
    check_architecture()

    # Step 1: Upgrade pip
    run_pip(["install", "--upgrade", "pip"], "Upgrading pip")

    # Step 2: Install base libraries
    base_pkgs = [
        "streamlit>=1.20.0",
        "pillow>=9.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0"
    ]
    if not run_pip(["install"] + base_pkgs, "Installing core libraries (Streamlit, NumPy, Pillow, Pandas)"):
        print("[WARNING] Encountered warnings during core library installation.")

    # Step 3: Install TensorFlow with smart fallbacks
    tf_installed = False
    
    # Try standard tensorflow
    print("\n--> Checking TensorFlow installation...")
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow {tf.__version__} is already installed!")
        tf_installed = True
    except ImportError:
        pass

    if not tf_installed:
        print("Attempting standard: pip install tensorflow...")
        if run_pip(["install", "tensorflow>=2.10.0"], "Installing standard TensorFlow"):
            tf_installed = True

    if not tf_installed:
        print("\nAttempting lightweight fallback: pip install tensorflow-cpu...")
        if run_pip(["install", "tensorflow-cpu>=2.10.0"], "Installing TensorFlow-CPU"):
            tf_installed = True

    if not tf_installed:
        print("\nAttempting pre-release fallback (for newer Python 3.13/3.14): pip install --pre tensorflow...")
        if run_pip(["install", "--pre", "tensorflow"], "Installing TensorFlow (--pre)"):
            tf_installed = True

    if not tf_installed:
        print("\n" + "=" * 75)
        print("❌ [ERROR] Could not install TensorFlow automatically.")
        print("Possible causes:")
        print("1. Your Python version might be too new (e.g., Python 3.13 without pre-release wheels).")
        print("   Recommended: Install Python 3.10, 3.11, or 3.12 (64-bit).")
        print("2. Network connectivity issue reaching PyPI.")
        print("=" * 75)
        sys.exit(1)

    print("\n[SUCCESS] All dependencies are ready!")

if __name__ == "__main__":
    install_all()
