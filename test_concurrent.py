#!/usr/bin/env python3
"""Test script for concurrent processing functionality."""

import time
from concurrent_evaluator import ConcurrentOpenAIEvaluator
from model_evaluator import OpenAIEvaluator

def test_concurrent_vs_sequential():
    """Compare concurrent vs sequential processing speed."""
    
    print("🧪 Testing Concurrent vs Sequential Processing")
    print("=" * 60)
    
    model = "gpt-4o-mini"
    split = "test_filtered_6"
    sample_size = 8  # Small test
    prompt_types = ["direct", "chain_of_thought"]
    
    print(f"Test configuration:")
    print(f"  Model: {model}")
    print(f"  Split: {split}")
    print(f"  Sample size: {sample_size}")
    print(f"  Prompt types: {prompt_types}")
    print(f"  Total requests: {sample_size * len(prompt_types)}")
    
    print("\n" + "="*60)
    
    # Test 1: Sequential processing
    print("🔄 Testing Sequential Processing...")
    sequential_start = time.time()
    
    sequential_evaluator = OpenAIEvaluator(model)
    sequential_results = sequential_evaluator.evaluate_dataset(
        split=split,
        prompt_types=prompt_types,
        sample_size=sample_size,
        save_results=False
    )
    
    sequential_time = time.time() - sequential_start
    sequential_accuracy = sequential_results['summary']['overall_performance']['accuracy']
    sequential_calls = sequential_results['api_usage']['total_calls']
    
    print(f"✅ Sequential completed!")
    print(f"  Time: {sequential_time:.1f}s")
    print(f"  Accuracy: {sequential_accuracy:.3f}")
    print(f"  API calls: {sequential_calls}")
    print(f"  Speed: {sequential_calls/sequential_time:.2f} requests/second")
    
    print("\n" + "-"*60)
    
    # Test 2: Concurrent processing
    print("⚡ Testing Concurrent Processing...")
    concurrent_start = time.time()
    
    concurrent_evaluator = ConcurrentOpenAIEvaluator(
        model, 
        max_concurrent=5,  # Conservative for testing
        requests_per_minute=60  # Respect rate limits
    )
    concurrent_results = concurrent_evaluator.evaluate_dataset_concurrent(
        split=split,
        prompt_types=prompt_types,
        sample_size=sample_size,
        save_results=False
    )
    
    concurrent_time = time.time() - concurrent_start
    concurrent_accuracy = concurrent_results['summary']['overall_performance']['accuracy']
    concurrent_calls = concurrent_results['api_usage']['total_calls']
    concurrent_rps = concurrent_results['api_usage']['requests_per_second']
    
    print(f"✅ Concurrent completed!")
    print(f"  Time: {concurrent_time:.1f}s")
    print(f"  Accuracy: {concurrent_accuracy:.3f}")
    print(f"  API calls: {concurrent_calls}")
    print(f"  Speed: {concurrent_rps:.2f} requests/second")
    
    print("\n" + "="*60)
    print("📊 COMPARISON RESULTS")
    print("="*60)
    
    # Calculate improvements
    if sequential_time > 0:
        speedup = sequential_time / concurrent_time
        print(f"🚀 Speed improvement: {speedup:.2f}x faster")
        print(f"⏰ Time saved: {sequential_time - concurrent_time:.1f} seconds")
    
    print(f"🎯 Accuracy difference: {abs(concurrent_accuracy - sequential_accuracy):.3f}")
    print(f"📞 API calls (both): {sequential_calls} = {concurrent_calls}")
    
    # Verify results are equivalent
    if abs(concurrent_accuracy - sequential_accuracy) < 0.1:
        print("✅ Results are equivalent!")
    else:
        print("⚠️  Significant accuracy difference detected")
    
    print(f"\n💡 Recommendations:")
    if speedup > 2:
        print(f"  ✅ Concurrent processing provides significant speedup!")
    elif speedup > 1.5:
        print(f"  ✅ Concurrent processing provides moderate speedup")
    else:
        print(f"  ⚠️  Limited speedup - may be rate limited")
    
    return {
        'sequential': {
            'time': sequential_time,
            'accuracy': sequential_accuracy,
            'calls': sequential_calls
        },
        'concurrent': {
            'time': concurrent_time,
            'accuracy': concurrent_accuracy,
            'calls': concurrent_calls,
            'requests_per_second': concurrent_rps
        },
        'speedup': speedup if sequential_time > 0 else 1
    }

def test_concurrent_configurations():
    """Test different concurrent configurations."""
    
    print("\n🔬 Testing Different Concurrent Configurations")
    print("=" * 60)
    
    configs = [
        {"max_concurrent": 3, "requests_per_minute": 30, "desc": "Conservative"},
        {"max_concurrent": 5, "requests_per_minute": 60, "desc": "Moderate"},
        {"max_concurrent": 8, "requests_per_minute": 100, "desc": "Aggressive"}
    ]
    
    model = "gpt-4o-mini"
    sample_size = 4
    prompt_types = ["direct"]
    
    results = []
    
    for config in configs:
        print(f"\n📋 Testing {config['desc']} configuration:")
        print(f"  Max concurrent: {config['max_concurrent']}")
        print(f"  Requests/minute: {config['requests_per_minute']}")
        
        start_time = time.time()
        
        evaluator = ConcurrentOpenAIEvaluator(
            model,
            max_concurrent=config['max_concurrent'],
            requests_per_minute=config['requests_per_minute']
        )
        
        result = evaluator.evaluate_dataset_concurrent(
            split="test_filtered_6",
            prompt_types=prompt_types,
            sample_size=sample_size,
            save_results=False
        )
        
        total_time = time.time() - start_time
        accuracy = result['summary']['overall_performance']['accuracy']
        rps = result['api_usage']['requests_per_second']
        
        print(f"  ✅ Time: {total_time:.1f}s, Accuracy: {accuracy:.3f}, RPS: {rps:.2f}")
        
        results.append({
            'config': config['desc'],
            'time': total_time,
            'accuracy': accuracy,
            'rps': rps
        })
    
    print(f"\n📊 Configuration Comparison:")
    for result in results:
        print(f"  {result['config']:12}: {result['time']:5.1f}s, {result['accuracy']:.3f} acc, {result['rps']:5.2f} rps")
    
    return results

def demonstrate_auto_selection():
    """Demonstrate automatic processing mode selection."""
    
    print("\n🤖 Testing Automatic Processing Mode Selection")
    print("=" * 60)
    
    from main import run_evaluation
    
    test_cases = [
        {"sample_size": 2, "expected": "sequential"},
        {"sample_size": 6, "expected": "concurrent"},
        {"sample_size": 15, "expected": "batch (but using concurrent in demo)"}
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}: {case['sample_size']} samples")
        print(f"Expected mode: {case['expected']}")
        print("-" * 40)
        
        # Run with auto-selection
        result = run_evaluation(
            model_name="gpt-4o-mini",
            split="test_filtered_6",
            prompt_types=["direct"],
            sample_size=case['sample_size'],
            use_batch=None,  # Let it auto-select
            use_concurrent=None  # Let it auto-select
        )
        
        print(f"✅ Test case {i} completed")

if __name__ == "__main__":
    print("🚀 Starting Concurrent Processing Tests")
    print("=" * 60)
    
    try:
        # Test 1: Speed comparison
        comparison_results = test_concurrent_vs_sequential()
        
        # Test 2: Different configurations
        config_results = test_concurrent_configurations()
        
        # Test 3: Auto-selection demo
        demonstrate_auto_selection()
        
        print(f"\n🎉 All tests completed successfully!")
        print(f"💡 Concurrent processing is ready for use!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 