#!/usr/bin/env python3
"""
Latency benchmark script for S2ST models.
"""

import argparse
import time
import numpy as np
import torch


def benchmark_pytorch(model_path: str, num_runs: int, audio_duration: float, device: str):
    """Benchmark PyTorch model."""
    print(f"\n=== PyTorch Benchmark ({device}) ===")
    
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    sample_rate = 16000
    audio = torch.randn(1, int(sample_rate * audio_duration)).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(audio)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(audio)
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    return np.array(latencies)


def benchmark_onnx(model_path: str, num_runs: int, audio_duration: float):
    """Benchmark ONNX model."""
    import onnxruntime as ort
    
    print(f"\n=== ONNX Benchmark ===")
    
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    sample_rate = 16000
    audio = np.random.randn(1, int(sample_rate * audio_duration)).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: audio})
    
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: audio})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    return np.array(latencies)


def benchmark_tflite(model_path: str, num_runs: int, audio_duration: float):
    """Benchmark TFLite model."""
    import tensorflow as tf
    
    print(f"\n=== TFLite Benchmark ===")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    sample_rate = 16000
    audio = np.random.randn(1, int(sample_rate * audio_duration)).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], audio)
        interpreter.invoke()
    
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], audio)
        interpreter.invoke()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    return np.array(latencies)


def print_results(latencies: np.ndarray, audio_duration: float):
    """Print benchmark results."""
    print(f"\nResults ({len(latencies)} runs):")
    print(f"  Audio duration: {audio_duration:.1f}s")
    print(f"  Mean latency:   {latencies.mean():.1f} ms")
    print(f"  Std latency:    {latencies.std():.1f} ms")
    print(f"  P50 latency:    {np.percentile(latencies, 50):.1f} ms")
    print(f"  P95 latency:    {np.percentile(latencies, 95):.1f} ms")
    print(f"  P99 latency:    {np.percentile(latencies, 99):.1f} ms")
    print(f"  Min latency:    {latencies.min():.1f} ms")
    print(f"  Max latency:    {latencies.max():.1f} ms")
    
    rtf = latencies.mean() / (audio_duration * 1000)
    print(f"\n  Real-time factor: {rtf:.3f}x")
    if rtf < 1.0:
        print(f"  ✓ Model is {1/rtf:.1f}x faster than real-time")
    else:
        print(f"  ✗ Model is {rtf:.1f}x slower than real-time")


def main():
    parser = argparse.ArgumentParser(description="Benchmark S2ST model latency")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--duration", type=float, default=5.0, help="Audio duration in seconds")
    parser.add_argument("--device", default="cuda", help="Device for PyTorch (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Detect model type and benchmark
    if args.model.endswith(".onnx"):
        latencies = benchmark_onnx(args.model, args.runs, args.duration)
    elif args.model.endswith(".tflite"):
        latencies = benchmark_tflite(args.model, args.runs, args.duration)
    else:
        latencies = benchmark_pytorch(args.model, args.runs, args.duration, args.device)
    
    print_results(latencies, args.duration)


if __name__ == "__main__":
    main()
