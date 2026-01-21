"""
Benchmark Comparison Script
Compares PyTorch, Phase 5 CUDA (custom kernels), and Phase 5.5 CUDA (cuBLAS) implementations
"""
import subprocess
import time
import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pytorch_mnist import train_mnist_pytorch

def run_phase5_cuda():
    """
    Run Phase 5 CUDA implementation (custom kernels) via subprocess
    
    Returns:
        tuple: (training_time, test_accuracy)
    """
    # Path to Phase 5 executable
    phase5_dir = Path(__file__).parent
    executable_path = phase5_dir / "build" / "phase5_neuron"
    
    if not executable_path.exists():
        # Try alternative path
        executable_path = phase5_dir / "phase5_neuron"
        if not executable_path.exists():
            raise FileNotFoundError(f"Phase 5 executable not found at {executable_path}")
    
    # Run the executable
    try:
        result = subprocess.run(
            [str(executable_path)],
            cwd=str(phase5_dir),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Parse output for training time and accuracy
        output = result.stdout + result.stderr
        print(output)
        
        import re
        training_time = None
        accuracy = None
        
        # Extract training time from output (ONLY the training time, not data loading/testing)
        # Phase 5 outputs: "GPU Training Time: X ms (Y.YY seconds)" or "Training Time: X.XX seconds"
        for line in output.split('\n'):
            # Look for "Training Time: X.XX seconds" (from FINAL RESULTS section)
            if 'Training Time:' in line and 'seconds' in line:
                match = re.search(r'Training Time:\s*(\d+\.?\d*)\s*seconds', line, re.IGNORECASE)
                if match:
                    training_time = float(match.group(1))
                    break
            # Also check for "GPU Training Time: X ms (Y.YY seconds)" format
            elif 'GPU Training Time:' in line and 'seconds' in line:
                match = re.search(r'\(\s*(\d+\.?\d*)\s*seconds\s*\)', line)
                if match:
                    training_time = float(match.group(1))
                    break
        
        if training_time is None:
            print("Warning: Could not parse training time from Phase 5 output")
            print("Expected format: 'Training Time: X.XX seconds' or 'GPU Training Time: X ms (Y.YY seconds)'")
        
        # Extract accuracy from output
        for line in output.split('\n'):
            if 'Accuracy:' in line or 'accuracy' in line.lower():
                match = re.search(r'(\d+\.?\d*)\s*%', line)
                if match:
                    accuracy = float(match.group(1))
                    break
        
        if accuracy is None:
            match = re.search(r'GPU\s+Accuracy:\s*(\d+\.?\d*)\s*%', output, re.IGNORECASE)
            if match:
                accuracy = float(match.group(1))
        
        if accuracy is None:
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if 'accuracy' in line.lower():
                    for j in range(i, min(i+3, len(lines))):
                        match = re.search(r'(\d+\.?\d*)\s*%', lines[j])
                        if match:
                            accuracy = float(match.group(1))
                            break
                    if accuracy:
                        break
        
        if accuracy is None:
            print("Warning: Could not parse accuracy from Phase 5 output")
            print("Output sample (last 20 lines):")
            print('\n'.join(output.split('\n')[-20:]))
            accuracy = 0.0
        
        if result.returncode != 0:
            print(f"Warning: Phase 5 executable returned non-zero exit code: {result.returncode}")
        
        print("-"*70)
        if training_time is not None and accuracy is not None:
            print(f"✓ Phase 5 CUDA completed: {training_time:.2f}s training time, {accuracy:.2f}% accuracy")
        elif training_time is not None:
            print(f"⚠ Phase 5 CUDA completed: {training_time:.2f}s training time (accuracy parsing failed)")
        else:
            print(f"✗ Phase 5 CUDA: Failed to parse training time from output")
        
        return training_time, accuracy
        
    except subprocess.TimeoutExpired:
        print("Error: Phase 5 execution timed out")
        return None, None
    except Exception as e:
        print(f"Error running Phase 5: {e}")
        return None, None

def run_phase5_5_cublas():
    """
    Run Phase 5.5 CUDA implementation (cuBLAS) via subprocess
    
    Returns:
        tuple: (training_time, test_accuracy)
    """
    # Path to Phase 5.5 executable
    phase5_dir = Path(__file__).parent
    executable_path = phase5_dir / "build" / "phase5_5_neuron"
    
    if not executable_path.exists():
        # Try alternative path
        executable_path = phase5_dir / "phase5_5_neuron"
        if not executable_path.exists():
            raise FileNotFoundError(f"Phase 5.5 executable not found at {executable_path}")
    
    # Run the executable
    try:
        result = subprocess.run(
            [str(executable_path)],
            cwd=str(phase5_dir),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Parse output for training time and accuracy
        output = result.stdout + result.stderr
        print(output)
        
        import re
        training_time = None
        accuracy = None
        
        # Extract training time from output (ONLY the training time, not data loading/testing)
        # Phase 5 outputs: "GPU Training Time: X ms (Y.YY seconds)" or "Training Time: X.XX seconds"
        # We want the seconds value, which is the actual training time
        for line in output.split('\n'):
            # Look for "Training Time: X.XX seconds" (from FINAL RESULTS section)
            if 'Training Time:' in line and 'seconds' in line:
                match = re.search(r'Training Time:\s*(\d+\.?\d*)\s*seconds', line, re.IGNORECASE)
                if match:
                    training_time = float(match.group(1))
                    break
            # Also check for "GPU Training Time: X ms (Y.YY seconds)" format
            elif 'GPU Training Time:' in line and 'seconds' in line:
                match = re.search(r'\(\s*(\d+\.?\d*)\s*seconds\s*\)', line)
                if match:
                    training_time = float(match.group(1))
                    break
        
        if training_time is None:
            print("Warning: Could not parse training time from Phase 5.5 output")
            print("Expected format: 'Training Time: X.XX seconds' or 'GPU Training Time: X ms (Y.YY seconds)'")
        
        # Extract accuracy from output
        for line in output.split('\n'):
            if 'Accuracy:' in line or 'accuracy' in line.lower():
                match = re.search(r'(\d+\.?\d*)\s*%', line)
                if match:
                    accuracy = float(match.group(1))
                    break
        
        if accuracy is None:
            match = re.search(r'GPU\s+Accuracy:\s*(\d+\.?\d*)\s*%', output, re.IGNORECASE)
            if match:
                accuracy = float(match.group(1))
        
        if accuracy is None:
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if 'accuracy' in line.lower():
                    for j in range(i, min(i+3, len(lines))):
                        match = re.search(r'(\d+\.?\d*)\s*%', lines[j])
                        if match:
                            accuracy = float(match.group(1))
                            break
                    if accuracy:
                        break
        
        if accuracy is None:
            print("Warning: Could not parse accuracy from Phase 5.5 output")
            print("Output sample (last 20 lines):")
            print('\n'.join(output.split('\n')[-20:]))
            accuracy = 0.0
        
        if result.returncode != 0:
            print(f"Warning: Phase 5.5 executable returned non-zero exit code: {result.returncode}")
        
        print("-"*70)
        if training_time is not None and accuracy is not None:
            print(f"✓ Phase 5.5 cuBLAS completed: {training_time:.2f}s training time, {accuracy:.2f}% accuracy")
        elif training_time is not None:
            print(f"⚠ Phase 5.5 cuBLAS completed: {training_time:.2f}s training time (accuracy parsing failed)")
        else:
            print(f"✗ Phase 5.5 cuBLAS: Failed to parse training time from output")
        
        return training_time, accuracy
        
    except subprocess.TimeoutExpired:
        print("Error: Phase 5.5 execution timed out")
        return None, None
    except Exception as e:
        print(f"Error running Phase 5.5: {e}")
        return None, None

def main():
    """Main benchmark comparison"""
    print("="*70)
    print(" " * 15 + "MNIST Training Benchmark Comparison")
    print("="*70)
    print("Comparing: PyTorch vs Phase 5 CUDA vs Phase 5.5 cuBLAS")
    print("="*70)
    
    # Data paths (relative to Phase-5-Optimization)
    base_dir = Path(__file__).parent.parent
    train_images_path = str(base_dir / "train-images.idx3-ubyte")
    train_labels_path = str(base_dir / "train-labels.idx1-ubyte")
    test_images_path = str(base_dir / "t10k-images.idx3-ubyte")
    test_labels_path = str(base_dir / "t10k-labels.idx1-ubyte")
    
    results = {}
    
    # Check for initial weights
    initial_weights_path = str(Path(__file__).parent / "initial_weights")
    use_same_weights = os.path.exists(f"{initial_weights_path}_W1.bin")
    
    if use_same_weights:
        print("\n✓ Using same initial weights for all implementations (fair comparison)")
    else:
        print("\n⚠ Warning: Initial weights not found. Each implementation will use different random initialization.")
        print("  To ensure fair comparison, run: ./generate_weights (after building)")
        initial_weights_path = None
    
    # Run PyTorch
    print("\n" + "="*70)
    print(" " * 20 + "TEST 1/3: PyTorch Implementation")
    print("="*70)
    print("Framework: PyTorch (GPU-accelerated)")
    print("Status: Running...")
    print("-"*70)
    try:
        pytorch_time, pytorch_accuracy, _ = train_mnist_pytorch(
            train_images_path, train_labels_path,
            test_images_path, test_labels_path,
            initial_weights_path=initial_weights_path
        )
        results['PyTorch'] = {
            'time': pytorch_time,
            'accuracy': pytorch_accuracy
        }
        print("-"*70)
        print(f"✓ PyTorch completed: {pytorch_time:.2f}s, {pytorch_accuracy:.2f}% accuracy")
    except Exception as e:
        print("-"*70)
        print(f"✗ PyTorch failed: {e}")
        import traceback
        traceback.print_exc()
        results['PyTorch'] = {'time': None, 'accuracy': None}
    
    # Run Phase 5 CUDA
    print("\n" + "="*70)
    print(" " * 10 + "TEST 2/3: Phase 5 CUDA Implementation")
    print("="*70)
    print("Framework: Custom CUDA kernels (Full GPU acceleration)")
    print("Status: Running...")
    print("-"*70)
    try:
        cuda_time, cuda_accuracy = run_phase5_cuda()
        results['Phase 5 CUDA'] = {
            'time': cuda_time,
            'accuracy': cuda_accuracy
        }
    except Exception as e:
        print("-"*70)
        print(f"✗ Phase 5 CUDA failed: {e}")
        import traceback
        traceback.print_exc()
        results['Phase 5 CUDA'] = {'time': None, 'accuracy': None}
    
    # Run Phase 5.5 cuBLAS
    print("\n" + "="*70)
    print(" " * 5 + "TEST 3/3: Phase 5.5 CUDA Implementation (cuBLAS)")
    print("="*70)
    print("Framework: NVIDIA cuBLAS (Optimized matrix multiplication)")
    print("Status: Running...")
    print("-"*70)
    try:
        cublas_time, cublas_accuracy = run_phase5_5_cublas()
        results['Phase 5.5 cuBLAS'] = {
            'time': cublas_time,
            'accuracy': cublas_accuracy
        }
    except Exception as e:
        print("-"*70)
        print(f"✗ Phase 5.5 cuBLAS failed: {e}")
        import traceback
        traceback.print_exc()
        results['Phase 5.5 cuBLAS'] = {'time': None, 'accuracy': None}
    
    # Print comparison table
    print("\n" + "="*70)
    print(" " * 20 + "FINAL RESULTS")
    print("="*70)
    print(f"{'Implementation':<22} {'Time (s)':<12} {'Accuracy (%)':<15} {'Speedup':<12}")
    print("-" * 70)
    
    pytorch_time = results['PyTorch']['time']
    
    for name, data in results.items():
        time_str = f"{data['time']:.2f}" if data['time'] is not None else "N/A"
        acc_str = f"{data['accuracy']:.2f}" if data['accuracy'] is not None else "N/A"
        
        if data['time'] is not None and pytorch_time is not None and pytorch_time > 0:
            speedup = pytorch_time / data['time']
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{name:<22} {time_str:<12} {acc_str:<15} {speedup_str:<12}")
    
    print("="*60)
    
    # Additional analysis
    if all(r['time'] is not None for r in results.values()):
        times = [r['time'] for r in results.values()]
        fastest_time = min(times)
        fastest_name = [name for name, data in results.items() if data['time'] == fastest_time][0]
        
        print(f"\nFastest Implementation: {fastest_name} ({fastest_time:.2f}s)")
        
        if pytorch_time and pytorch_time > 0:
            print(f"\nSpeedup Analysis (relative to PyTorch):")
            for name, data in results.items():
                if data['time'] and name != 'PyTorch':
                    speedup = pytorch_time / data['time']
                    print(f"  {name}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    if all(r['accuracy'] is not None for r in results.values()):
        accuracies = [r['accuracy'] for r in results.values()]
        print(f"\nAccuracy Range: {min(accuracies):.2f}% - {max(accuracies):.2f}%")
        accuracy_diff = max(accuracies) - min(accuracies)
        print(f"Accuracy Difference: {accuracy_diff:.2f}%")
        
        if accuracy_diff < 1.0:
            print("✓ All implementations achieved similar accuracy!")
        else:
            print("⚠ Accuracy differs significantly between implementations")

if __name__ == "__main__":
    main()
