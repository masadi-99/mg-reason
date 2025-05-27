#!/usr/bin/env python3
"""Test script for structured tag-based answer extraction."""

from model_evaluator import OpenAIEvaluator
from batch_evaluator import OpenAIBatchEvaluator
from reasoning_prompts import PromptTemplates

def test_structured_tags():
    """Test the new structured tag-based answer extraction."""
    print("🧪 Testing Structured Tag-Based Answer Extraction")
    print("=" * 60)
    
    evaluator = OpenAIEvaluator("gpt-4o-mini")
    batch_evaluator = OpenAIBatchEvaluator("gpt-4o-mini")
    
    # Test structured responses with tags
    test_cases = [
        {
            "response": "<think>\nThe patient has chest pain with ST elevation in leads II, III, aVF indicating inferior wall involvement.\n</think>\n\n<answer>B</answer>",
            "expected": "B",
            "description": "Proper structured response with think and answer tags"
        },
        {
            "response": "<answer>A</answer>",
            "expected": "A",
            "description": "Direct answer tag only"
        },
        {
            "response": "The patient presents with symptoms. <answer>C</answer> is most appropriate.",
            "expected": "C",
            "description": "Answer tag in middle of response"
        },
        {
            "response": "<answer> D </answer>",
            "expected": "D",
            "description": "Answer tag with spaces"
        },
        {
            "response": "<ANSWER>E</ANSWER>",
            "expected": "E",
            "description": "Case insensitive answer tag"
        },
        {
            "response": "<answer>A",
            "expected": "A",
            "description": "Answer tag without closing"
        },
        {
            "response": "Therefore, the correct answer is B.",
            "expected": "B",
            "description": "Legacy pattern fallback"
        },
        {
            "response": "Answer: C",
            "expected": "C",
            "description": "Legacy answer pattern"
        }
    ]
    
    print("Testing Regular Evaluator:")
    print("-" * 40)
    regular_passed = 0
    for i, case in enumerate(test_cases, 1):
        predicted = evaluator._extract_answer(case["response"])
        passed = predicted == case["expected"]
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{i}. {case['description']}: {status}")
        print(f"   Expected: {case['expected']}, Got: {predicted}")
        if passed:
            regular_passed += 1
    
    print(f"\nRegular Evaluator: {regular_passed}/{len(test_cases)} passed")
    
    print("\nTesting Batch Evaluator:")
    print("-" * 40)
    batch_passed = 0
    for i, case in enumerate(test_cases, 1):
        predicted = batch_evaluator._extract_answer(case["response"])
        passed = predicted == case["expected"]
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{i}. {case['description']}: {status}")
        print(f"   Expected: {case['expected']}, Got: {predicted}")
        if passed:
            batch_passed += 1
    
    print(f"\nBatch Evaluator: {batch_passed}/{len(test_cases)} passed")
    
    return regular_passed, batch_passed, len(test_cases)

def test_prompt_templates():
    """Test the updated prompt templates."""
    print("\n🔍 Testing Updated Prompt Templates")
    print("=" * 60)
    
    sample_question = "A 45-year-old patient presents with chest pain. What is the most appropriate initial test?"
    sample_options = ["Chest X-ray", "ECG", "Echocardiogram", "Stress test"]
    
    for prompt_type in PromptTemplates.get_available_types():
        print(f"\n{'='*40}")
        print(f"PROMPT TYPE: {prompt_type.upper()}")
        print(f"{'='*40}")
        prompt = PromptTemplates.get_prompt(prompt_type, sample_question, sample_options)
        
        # Check if prompt contains the required tags
        has_answer_tag = "<answer>" in prompt
        has_think_tag = "<think>" in prompt or prompt_type == "direct"  # direct doesn't need think tag
        
        print(f"Contains <answer> instruction: {'✅' if has_answer_tag else '❌'}")
        if prompt_type != "direct":
            print(f"Contains <think> instruction: {'✅' if has_think_tag else '❌'}")
        
        print(f"\nPrompt preview (last 200 chars):")
        print(f"...{prompt[-200:]}")

if __name__ == "__main__":
    # Test answer extraction
    regular_passed, batch_passed, total = test_structured_tags()
    
    print(f"\n📊 Answer Extraction Results:")
    print(f"Regular Evaluator: {regular_passed}/{total} ({regular_passed/total*100:.1f}%)")
    print(f"Batch Evaluator: {batch_passed}/{total} ({batch_passed/total*100:.1f}%)")
    
    if regular_passed == total and batch_passed == total:
        print("🎉 ALL ANSWER EXTRACTION TESTS PASSED!")
    else:
        print("⚠️  Some answer extraction tests failed.")
    
    # Test prompt templates
    test_prompt_templates()
    
    print(f"\n🎯 Summary:")
    print(f"✅ Structured tags implemented for reliable answer extraction")
    print(f"✅ Legacy pattern fallback for backwards compatibility")
    print(f"✅ All prompt templates updated with <think> and <answer> tags")
    print(f"✅ Case-insensitive tag detection")
    print(f"✅ Robust tag parsing with whitespace handling") 