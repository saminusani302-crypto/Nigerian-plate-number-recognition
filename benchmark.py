"""
Performance Benchmarking Script for ALPR System
Measures and reports system performance metrics
"""

import time
import cv2
import numpy as np
from pathlib import Path
from alpr_system import ALPRPipeline
import json


def benchmark_detector(alpr, num_frames=50):
    """Benchmark detection performance."""
    print("\n" + "="*60)
    print("DETECTOR BENCHMARK")
    print("="*60)
    
    # Create random frames
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(num_frames)]
    
    # Warmup
    alpr.detector.detect_objects(frames[0])
    
    # Benchmark
    times = []
    for frame in frames:
        start = time.time()
        alpr.detector.detect_objects(frame)
        times.append(time.time() - start)
    
    return {
        'avg_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': 1.0 / np.mean(times)
    }


def benchmark_preprocessor(alpr, num_frames=50):
    """Benchmark preprocessing performance."""
    print("\n" + "="*60)
    print("PREPROCESSOR BENCHMARK")
    print("="*60)
    
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(num_frames)]
    
    # Warmup
    alpr.preprocessor.preprocess_for_detection(frames[0])
    
    # Benchmark
    times = []
    for frame in frames:
        start = time.time()
        alpr.preprocessor.preprocess_for_detection(frame)
        times.append(time.time() - start)
    
    return {
        'avg_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': 1.0 / np.mean(times)
    }


def benchmark_pipeline(alpr, num_frames=50):
    """Benchmark complete pipeline."""
    print("\n" + "="*60)
    print("PIPELINE BENCHMARK")
    print("="*60)
    
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(num_frames)]
    
    # Warmup
    alpr.process_frame(frames[0], log_results=False)
    
    # Benchmark
    times = []
    for frame in frames:
        start = time.time()
        alpr.process_frame(frame, log_results=False)
        times.append(time.time() - start)
    
    return {
        'avg_time': np.mean(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': 1.0 / np.mean(times)
    }


def print_results(name, results):
    """Print benchmark results."""
    print(f"\n{name}:")
    print(f"  Average time: {results['avg_time']*1000:.2f} ms")
    print(f"  Min time: {results['min_time']*1000:.2f} ms")
    print(f"  Max time: {results['max_time']*1000:.2f} ms")
    print(f"  FPS: {results['fps']:.2f}")


def run_benchmarks(device='cpu', num_frames=50):
    """Run all benchmarks."""
    print("\n" + "#"*60)
    print("# NIGERIAN ALPR SYSTEM - PERFORMANCE BENCHMARK")
    print("#"*60)
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Frames: {num_frames}")
    print(f"  Model: yolov8n.pt")
    
    # Initialize pipeline
    print("\nInitializing ALPR Pipeline...")
    alpr = ALPRPipeline(device=device, confidence_threshold=0.5)
    
    # Run benchmarks
    benchmark_results = {}
    
    try:
        preprocessor_results = benchmark_preprocessor(alpr, num_frames)
        benchmark_results['preprocessor'] = preprocessor_results
        print_results("PREPROCESSOR", preprocessor_results)
    except Exception as e:
        print(f"Preprocessor benchmark failed: {e}")
    
    try:
        detector_results = benchmark_detector(alpr, num_frames)
        benchmark_results['detector'] = detector_results
        print_results("DETECTOR", detector_results)
    except Exception as e:
        print(f"Detector benchmark failed: {e}")
    
    try:
        pipeline_results = benchmark_pipeline(alpr, num_frames)
        benchmark_results['pipeline'] = pipeline_results
        print_results("PIPELINE", pipeline_results)
    except Exception as e:
        print(f"Pipeline benchmark failed: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if 'pipeline' in benchmark_results:
        print(f"\nEnd-to-End Performance:")
        print(f"  Frames Processed: {num_frames}")
        print(f"  Average Time per Frame: {benchmark_results['pipeline']['avg_time']*1000:.2f} ms")
        print(f"  Throughput: {benchmark_results['pipeline']['fps']:.2f} FPS")
    
    # Save results
    results_file = f'benchmark_results_{device}_{int(time.time())}.json'
    with open(results_file, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return benchmark_results


def compare_devices():
    """Compare CPU vs GPU performance."""
    print("\n" + "#"*60)
    print("# DEVICE COMPARISON BENCHMARK")
    print("#"*60)
    
    devices = ['cpu']
    
    # Check if CUDA available
    try:
        import torch
        if torch.cuda.is_available():
            devices.append('cuda')
    except:
        pass
    
    results = {}
    for device in devices:
        print(f"\n{'='*60}")
        print(f"Testing on {device.upper()}")
        print(f"{'='*60}")
        results[device] = run_benchmarks(device=device, num_frames=30)
    
    # Comparison
    if len(results) > 1:
        print("\n" + "#"*60)
        print("# DEVICE COMPARISON")
        print("#"*60)
        
        cpu_fps = results['cpu']['pipeline']['fps']
        
        for device, res in results.items():
            if device != 'cpu':
                device_fps = res['pipeline']['fps']
                speedup = device_fps / cpu_fps
                print(f"\n{device.upper()} Speedup vs CPU: {speedup:.2f}x")


def memory_usage():
    """Check memory usage."""
    print("\n" + "="*60)
    print("MEMORY USAGE")
    print("="*60)
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        print(f"  RSS Memory: {mem_info.rss / 1024 / 1024:.2f} MB")
        print(f"  VMS Memory: {mem_info.vms / 1024 / 1024:.2f} MB")
    except ImportError:
        print("  Install psutil for memory monitoring: pip install psutil")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ALPR System Benchmarking')
    parser.add_argument('--device', default='cpu', help='Device: cpu or cuda')
    parser.add_argument('--frames', type=int, default=50, help='Number of frames to test')
    parser.add_argument('--compare', action='store_true', help='Compare CPU vs GPU')
    parser.add_argument('--memory', action='store_true', help='Check memory usage')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_devices()
    else:
        run_benchmarks(device=args.device, num_frames=args.frames)
    
    if args.memory:
        memory_usage()
