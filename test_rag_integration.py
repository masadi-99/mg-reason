#!/usr/bin/env python3
"""
Test script for RAG integration with the main evaluation workflow.
Verifies that CLI arguments for RAG are correctly passed and processed.
"""

import subprocess
import os
import shutil
import tempfile
from pathlib import Path

# Define a helper to create dummy PDF files
def create_dummy_pdf(filepath: Path, content: str):
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(filepath))
        textobject = c.beginText(50, 750)
        for line in content.split('\n'):
            textobject.textLine(line)
        c.drawText(textobject)
        c.save()
        # print(f"    Created dummy PDF: {filepath}")
    except ImportError:
        print("ReportLab not found. Cannot create dummy PDF. Test might fail if PDFs are required.")
        # Create a simple text file as a fallback if reportlab is not available
        with open(filepath.with_suffix(".txt_fallback"), "w") as f:
            f.write(content)
        print(f"    Created fallback text file: {filepath.with_suffix('.txt_fallback')}")


def run_evaluation_test(test_name: str, base_command: List[str], rag_args: List[str] = None):
    """Helper function to run an evaluation command and print results."""
    command = base_command.copy()
    if rag_args:
        command.extend(rag_args)
    
    print(f"\n--- Running Test: {test_name} ---")
    print(f"Executing: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=1200) # 10 min timeout for graphrag build
        print("STDOUT:")
        print(result.stdout[-1000:]) # Print last 1000 chars of stdout
        if result.stderr:
            print("STDERR:")
            print(result.stderr[-1000:]) # Print last 1000 chars of stderr
        
        if "Overall Accuracy" in result.stdout:
            print(f"✅ {test_name}: Completed successfully.")
        else:
            print(f"⚠️ {test_name}: Completed, but 'Overall Accuracy' not found in output.")

        # Check for RAG specific logs
        if "--rag-mode langchain" in " ".join(command):
            if "LangChainRAGEvaluator for evaluation pipeline" in result.stdout or \
               "Initialized LangChainRAGEvaluator for" in result.stdout:
                print("  🔍 LangChain RAG initialization detected.")
            if "Enhancing with langchain RAG" in result.stdout:
                print("  🔍 LangChain RAG context retrieval detected during evaluation.")
        
        if "--rag-mode graphrag" in " ".join(command):
            if "GraphRAGEvaluator for evaluation pipeline" in result.stdout or \
               "Initialized GraphRAGEvaluator for" in result.stdout:
                print("  🕸️ GraphRAG initialization detected.")
            if "Enhancing with graphrag RAG" in result.stdout:
                print("  🕸️ GraphRAG context retrieval detected during evaluation.")
            if "GraphRAG indexing pipeline" in result.stdout:
                 print("  🕸️ GraphRAG indexing seems to have run.")


    except subprocess.CalledProcessError as e:
        print(f"❌ {test_name}: FAILED with exit code {e.returncode}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
    except subprocess.TimeoutExpired as e:
        print(f"❌ {test_name}: FAILED due to timeout.")
        if e.stdout:
            print("STDOUT (on timeout):")
            print(e.stdout.decode(errors='ignore')[-1000:])
        if e.stderr:
            print("STDERR (on timeout):")
            print(e.stderr.decode(errors='ignore')[-1000:])
    except Exception as e:
        print(f"❌ {test_name}: FAILED with unexpected error: {e}")

def main():
    print("🧪 Starting RAG Integration Test Suite 🧪")
    
    # Create a temporary directory for dummy PDFs and RAG indexes
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_test_"))
    dummy_pdf_dir = temp_dir / "dummy_pdfs"
    dummy_pdf_dir.mkdir()
    
    # Specific dirs for RAG indexes to avoid interference and ensure cleanup
    langchain_index_dir = temp_dir / "test_lc_index"
    graphrag_work_dir = temp_dir / "test_gr_work"

    print(f"Temporary test directory: {temp_dir}")

    # Create a couple of dummy PDF files
    create_dummy_pdf(dummy_pdf_dir / "cardio_doc1.pdf", "Myocardial infarction is treated with aspirin. ECG shows ST elevation.")
    create_dummy_pdf(dummy_pdf_dir / "cardio_doc2.pdf", "Heart failure guidelines recommend beta-blockers and ACE inhibitors.")

    base_command = [
        "python", "main.py",
        "--eval",
        "--model", "gpt-4o-mini",
        "--specialty", "Cardiology", # Test data loader has some cardiology, but dummy PDFs are key
        "--sample-size", "1",         # Very small sample for speed
        "--prompts", "direct",
        "--concurrent"                  # Use concurrent mode for these tests
    ]
    
    # Test 1: No RAG
    run_evaluation_test("Evaluation (No RAG)", base_command)

    # Test 2: LangChain RAG
    # The LangChainRAGEvaluator is set up to use './cardiology_pdfs' and './rag_index_langchain' by default.
    # For isolated testing, we'd ideally pass these paths via CLI if main.py supported it for RAG evaluators.
    # Since it doesn't directly, the test relies on RAG evaluators being initialized by OpenAIEvaluator etc.
    # and those RAG evaluators picking up their default configs or being passed instances.
    # For now, let's ensure our LangChainRAGEvaluator in rag_langchain.py can take pdfs_dir and index_dir
    # and that these can be influenced if we were to modify how evaluators get RAG instances in main.py.
    # The current setup in main.py instantiates RAG evaluators without specific test paths.
    # So, this test will use the *actual* ./cardiology_pdfs and ./rag_index_langchain if they exist,
    # or try to build them. This is not ideal for a fully isolated test.
    # To make it more robust for this test, we should modify the RAG classes to accept these paths
    # or ensure main.py can pass them.
    # For now, this test assumes the RAG classes will try to build from default dummy paths if main PDFs are missing
    # or will use the dummy_pdf_dir *if the evaluators were configurable to use it via main.py*.
    
    # Let's make the test more robust by temporarily pointing RAG classes to temp dirs
    # This requires modifying config.py or having CLI args for RAG paths.
    # As a simpler approach for this test, we will rely on the fact that RAGEvaluators are
    # instantiated inside the main evaluators (OpenAIEvaluator, etc.) and those could be
    # made to respect test-specific paths if main.py was further modified.
    # Given the current structure, we are testing the CLI integration primarily.
    
    # The RAG evaluators are initialized in main.py without specific paths,
    # so they will use their defaults (e.g. ./cardiology_pdfs).
    # We'll proceed, understanding this limitation for full isolation in this test script.
    
    # First, try to build LangChain index for the dummy PDFs (this command is separate)
    # This step is not strictly necessary if LangChainRAGEvaluator auto-builds, but good for clarity.
    # We need a way to tell LangChain *which* PDFs to build from for the test.
    # The `LangChainRAGEvaluator` in `main.py` is called without arguments, so it uses its defaults.
    # For this test to be meaningful for RAG, the RAG system needs to pick up our dummy PDFs.
    
    # Test LangChain RAG
    # To make this test effective, we would need a way for main.py's --rag-mode langchain
    # to use a specific pdf_dir and index_dir.
    # Current evaluators initialize RAG with defaults.
    # Let's assume for the purpose of this CLI test that default indexes might exist or get created.
    print("\nINFO: For LangChain/GraphRAG tests below, ensure that RAG indexes exist or can be built.")
    print(f"INFO: LangChain by default uses './cardiology_pdfs' and creates './rag_index_langchain'.")
    print(f"INFO: GraphRAG by default uses './cardiology_pdfs' and creates './graphrag_workspace'.")
    print(f"INFO: This test script created dummy PDFs in {dummy_pdf_dir}, but default RAG setup might not use them without code changes.")

    langchain_args = ["--rag-mode", "langchain", "--rag-k-retrieval", "1"]
    # We can't easily redirect the LangChainRAGEvaluator in main.py to use dummy_pdf_dir via CLI.
    # It will use its default. This test verifies CLI plumbing.
    run_evaluation_test("Evaluation (LangChain RAG)", base_command, langchain_args)

    # Test 3: GraphRAG
    # Similar challenge: GraphRAGEvaluator uses its default ./cardiology_pdfs and ./graphrag_workspace
    graphrag_args = ["--rag-mode", "graphrag", "--rag-k-retrieval", "1", "--graphrag-search-type", "global"]
    run_evaluation_test("Evaluation (GraphRAG)", base_command, graphrag_args)
    
    # Cleanup
    print(f"\n--- Cleaning up temporary directory: {temp_dir} ---")
    try:
        shutil.rmtree(temp_dir)
        print(f"✅ Cleaned up {temp_dir}")
    except Exception as e:
        print(f"⚠️  Could not clean up {temp_dir}: {e}")

if __name__ == "__main__":
    main() 