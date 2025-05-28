#!/usr/bin/env python3
"""
Test script for Adaptive RAG functionality.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from adaptive_rag_evaluator import AdaptiveRAGEvaluator

def test_adaptive_rag():
    """Test the adaptive RAG evaluator on a single sample."""
    print("🧪 Testing Adaptive RAG Evaluator")
    print("=" * 50)
    
    # Create a test sample
    test_sample = {
        'Question': 'A 65-year-old man with a history of myocardial infarction presents with shortness of breath and ankle swelling. His ejection fraction is 35%. Which of the following medications would be most appropriate as first-line therapy?',
        'Options': [
            'Digoxin',
            'ACE inhibitor',
            'Calcium channel blocker',
            'Thiazide diuretic'
        ],
        'Answer': 'ACE inhibitor',
        'Specialty': 'Cardiology'
    }
    
    try:
        # Initialize evaluator
        evaluator = AdaptiveRAGEvaluator(
            model_name="gpt-4o-mini",
            k_guidelines=2,  # Use 2 guidelines for faster testing
            k_retrieval_per_guideline=1  # Use 1 retrieval per guideline for faster testing
        )
        
        print("✅ Evaluator initialized successfully")
        
        # Test single sample evaluation
        print("\n🔍 Testing single sample evaluation...")
        result = evaluator.evaluate_sample_adaptive(test_sample, "direct")
        
        print(f"\n📊 Results:")
        print(f"Question: {result['question'][:100]}...")
        print(f"Predicted: {result['predicted_choice']}")
        print(f"Correct: {result['correct_choice']}")
        print(f"Is Correct: {result['is_correct']}")
        print(f"Total Time: {result['total_response_time']:.2f}s")
        
        print(f"\n📋 Identified Guidelines:")
        for i, guideline in enumerate(result['stage1_identified_guidelines'], 1):
            print(f"  {i}. {guideline[:100]}...")
        
        print(f"\n⏱️  Stage Breakdown:")
        print(f"  Stage 1 (Guidelines): {result['stage1_time']:.2f}s")
        print(f"  Stage 2 (Retrieval): {result['stage2_time']:.2f}s")
        print(f"  Stage 3 (Answer): {result['stage3_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_adaptive_rag()
    if success:
        print("\n✅ Adaptive RAG test completed successfully!")
    else:
        print("\n❌ Adaptive RAG test failed!")
        sys.exit(1) 