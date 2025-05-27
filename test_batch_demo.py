#!/usr/bin/env python3
"""Demo test script for batch processing functionality."""

from main import run_evaluation
from config import BATCH_SETTINGS

print("🎭 Testing Batch Processing Demo Mode")
print("=" * 50)
print("This demonstrates batch processing without the long wait times!")

# Test 1: Enable demo mode temporarily
print("\n1️⃣ Enabling demo mode...")
original_demo = BATCH_SETTINGS["demo_mode"]
BATCH_SETTINGS["demo_mode"] = True

try:
    # Test 2: Run batch evaluation in demo mode
    print("\n2️⃣ Running batch evaluation demo...")
    results = run_evaluation(
        model_name="gpt-4o-mini",
        split="test_filtered_6",
        prompt_types=["direct", "chain_of_thought"],
        sample_size=10,
        use_batch=True
    )
    
    # Test 3: Verify demo results
    print("\n3️⃣ Verifying demo results...")
    processing_mode = "Batch (Demo)" if results['api_usage'].get('batch_processing') else "Synchronous"
    print(f"   ✅ Evaluation completed using {processing_mode}")
    print(f"   📊 Accuracy: {results['summary']['overall_performance']['accuracy']:.3f}")
    print(f"   💰 Simulated cost savings: ~50%")
    
finally:
    # Restore original demo mode
    BATCH_SETTINGS["demo_mode"] = original_demo
    print(f"\n🔄 Restored demo mode to: {original_demo}")

print("\n🎉 Demo completed successfully!")
print("\n💡 To use real batch processing:")
print("   • Set demo_mode=False in config.py")
print("   • Be prepared to wait 10 minutes - 24 hours")
print("   • Get 50% cost savings on API calls")
print("   • Process hundreds/thousands of samples concurrently") 