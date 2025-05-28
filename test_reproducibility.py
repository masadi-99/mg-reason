#!/usr/bin/env python3
"""
Test script to verify reproducible evaluation results.
This test runs the same evaluation twice and checks if results are identical.
"""

import subprocess
import json
import os
import tempfile
from pathlib import Path

def run_evaluation_command(cmd_args, output_file):
    """Run evaluation command and capture results."""
    # Add redirect to save results to a specific file
    env = os.environ.copy()
    result = subprocess.run(
        cmd_args,
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return None
    
    return result.stdout

def extract_results_from_output(output_text):
    """Extract key results from the command output."""
    lines = output_text.split('\n')
    results = {}
    
    for line in lines:
        if 'accuracy:' in line.lower():
            results['accuracy_line'] = line.strip()
        elif 'correct_count' in line.lower():
            results['correct_count_line'] = line.strip()
        elif 'Total API calls:' in line:
            results['api_calls'] = line.strip()
        elif 'Total tokens used:' in line:
            results['tokens'] = line.strip()
    
    return results

def main():
    print("🔬 Testing reproducibility of evaluation results...")
    
    # Test command - using a small sample for quick testing
    base_cmd = [
        "python", "main.py", 
        "--eval", 
        "--specialty", "Cardiology", 
        "--sample-size", "5",  # Small sample for quick test
        "--model", "gpt-4o-mini", 
        "--concurrent", 
        "--prompts", "direct"
    ]
    
    # Test 1: Basic evaluation
    print("\\n📊 Test 1: Basic evaluation reproducibility")
    print("Running first evaluation...")
    output1 = run_evaluation_command(base_cmd, "test_output1.txt")
    
    print("Running second evaluation...")
    output2 = run_evaluation_command(base_cmd, "test_output2.txt")
    
    if output1 and output2:
        results1 = extract_results_from_output(output1)
        results2 = extract_results_from_output(output2)
        
        print("\\nResults comparison:")
        for key in results1:
            if key in results2:
                match = results1[key] == results2[key]
                status = "✅ MATCH" if match else "❌ DIFFER"
                print(f"  {key}: {status}")
                if not match:
                    print(f"    Run 1: {results1[key]}")
                    print(f"    Run 2: {results2[key]}")
        
        # Overall check
        all_match = all(results1.get(k) == results2.get(k) for k in results1 if k in results2)
        print(f"\\n🎯 Basic evaluation: {'✅ REPRODUCIBLE' if all_match else '❌ NOT REPRODUCIBLE'}")
    else:
        print("❌ Failed to run basic evaluation commands")
    
    # Test 2: RAG evaluation (if LangChain index exists)
    rag_index_path = Path("./rag_index_langchain")
    if rag_index_path.exists():
        print("\\n📊 Test 2: RAG evaluation reproducibility")
        rag_cmd = base_cmd + [
            "--rag-mode", "langchain", 
            "--rag-k-retrieval", "3"
        ]
        
        print("Running first RAG evaluation...")
        rag_output1 = run_evaluation_command(rag_cmd, "test_rag_output1.txt")
        
        print("Running second RAG evaluation...")
        rag_output2 = run_evaluation_command(rag_cmd, "test_rag_output2.txt")
        
        if rag_output1 and rag_output2:
            rag_results1 = extract_results_from_output(rag_output1)
            rag_results2 = extract_results_from_output(rag_output2)
            
            print("\\nRAG Results comparison:")
            for key in rag_results1:
                if key in rag_results2:
                    match = rag_results1[key] == rag_results2[key]
                    status = "✅ MATCH" if match else "❌ DIFFER"
                    print(f"  {key}: {status}")
                    if not match:
                        print(f"    Run 1: {rag_results1[key]}")
                        print(f"    Run 2: {rag_results2[key]}")
            
            # Overall check
            rag_all_match = all(rag_results1.get(k) == rag_results2.get(k) for k in rag_results1 if k in rag_results2)
            print(f"\\n🎯 RAG evaluation: {'✅ REPRODUCIBLE' if rag_all_match else '❌ NOT REPRODUCIBLE'}")
        else:
            print("❌ Failed to run RAG evaluation commands")
    else:
        print("\\n⏭️  Skipping RAG test (no LangChain index found)")
    
    print("\\n🏁 Reproducibility test completed!")
    print("\\nNote: If evaluations are not reproducible, check:")
    print("  - OpenAI API parameters (temperature, seed, etc.)")
    print("  - Data sampling randomness")
    print("  - RAG retrieval consistency")

if __name__ == "__main__":
    main() 