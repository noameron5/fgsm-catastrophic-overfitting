"""
Dependency checker and installer for the catastrophic overfitting demo.

This script checks if required libraries are installed and installs them if needed.
"""

import importlib
import subprocess
import sys


def check_and_install(lib):
    """
    Check if a library is installed, and install it if not.
    
    Args:
        lib (str): Library name to check/install
    """
    # Handle version specifiers
    lib_name = lib.split('<')[0].split('>')[0].split('=')[0]
    
    try:
        importlib.import_module(lib_name)
        print(f"✓ {lib} is already installed.")
    except ImportError:
        print(f"✗ {lib} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"✓ {lib} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing {lib}: {e}")
            sys.exit(1)


def main():
    """Main function to check and install all required libraries."""
    libraries = [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy<2.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.12.0',
    ]
    
    print("="*80)
    print("Checking and installing dependencies...")
    print("="*80)
    
    for lib in libraries:
        check_and_install(lib)
    
    print("\n" + "="*80)
    print("All dependencies are installed!")
    print("="*80)


if __name__ == "__main__":
    main()
