#!/usr/bin/env python3
"""Main script for medical reasoning evaluation system."""
import argparse
import json
import os
from typing import List, Optional, Dict

from config import OPENAI_MODELS, EVALUATION_SETTINGS, BATCH_SETTINGS, CONCURRENT_CONFIG
from data_loader import MedQADataLoader
from model_evaluator import OpenAIEvaluator
from batch_evaluator import OpenAIBatchEvaluator, check_batch_status, list_batch_jobs
from reasoning_prompts import PromptTemplates
from bmj_pdf_downloader import BMJPDFDownloader
from concurrent_evaluator import ConcurrentOpenAIEvaluator
from rag_langchain import LangChainRAGEvaluator
from rag_graphrag import GraphRAGEvaluator
from adaptive_rag_evaluator import AdaptiveRAGEvaluator

def print_banner():
    """Print a welcome banner."""
    print("=" * 60)
    print("🏥 Medical Reasoning Evaluation System")
    print("   Exploring AI Performance on Medical Datasets")
    print("   🚀 Now with Batch Processing Support!")
    print("=" * 60)

def analyze_dataset():
    """Analyze and display dataset statistics."""
    print("\n📊 Dataset Analysis")
    print("-" * 40)
    
    loader = MedQADataLoader()
    stats = loader.get_dataset_stats()
    
    for split, stat in stats.items():
        print(f"\n{split.upper()} SPLIT:")
        print(f"  📋 Total samples: {stat['total_samples']:,}")
        print(f"  📝 Avg question length: {stat['avg_question_length']:.1f} characters")
        print(f"  🏥 Top 5 specialties:")
        for specialty, count in list(stat['specialties'].items())[:5]:
            print(f"    • {specialty}: {count}")
    
    return loader

def run_evaluation(model_name: str, 
                  split: str = "test",
                  prompt_types: List[str] = ["direct"],
                  sample_size: Optional[int] = None,
                  specialty_filter: Optional[str] = None,
                  use_batch: bool = None,
                  use_concurrent: bool = None,
                  max_concurrent: int = None,
                  requests_per_minute: int = None,
                  rag_mode: Optional[str] = None,
                  rag_params: Optional[Dict] = None):
    """Run evaluation on specified configuration with auto processing mode selection and RAG."""
    
    print(f"\n🤖 Starting Evaluation")
    print("-" * 40)
    print(f"Model: {model_name}")
    print(f"Split: {split}")
    print(f"Prompt types: {', '.join(prompt_types)}")
    if sample_size:
        print(f"Sample size: {sample_size}")
    if specialty_filter:
        print(f"Specialty filter: {specialty_filter}")
    if rag_mode:
        print(f"🧠 RAG Mode: {rag_mode}")
        if rag_params:
            print(f"   RAG Params: {rag_params}")
    
    # Calculate total requests to determine processing mode
    data_loader = MedQADataLoader()
    sample_data = data_loader.get_split(split, sample_size, specialty_filter)
    total_requests = len(sample_data) * len(prompt_types)
    
    print(f"📊 Total requests: {total_requests}")
    
    # Determine processing mode with smart auto-selection
    processing_mode = None
    
    if use_batch is True:
        processing_mode = "batch"
    elif use_concurrent is True:
        processing_mode = "concurrent"
    elif use_batch is False and use_concurrent is False:
        processing_mode = "sequential"
    elif EVALUATION_SETTINGS['auto_concurrent']:
        # Auto-select based on request count
        if total_requests >= EVALUATION_SETTINGS['batch_threshold']:
            processing_mode = "batch"
            print(f"🚀 Auto-selecting batch processing (≥{EVALUATION_SETTINGS['batch_threshold']} requests)")
        elif total_requests >= EVALUATION_SETTINGS['concurrent_threshold']:
            processing_mode = "concurrent"
            print(f"⚡ Auto-selecting concurrent processing (≥{EVALUATION_SETTINGS['concurrent_threshold']} requests)")
        else:
            processing_mode = "sequential"
            print(f"🔄 Using sequential processing (<{EVALUATION_SETTINGS['concurrent_threshold']} requests)")
    else:
        # Fallback to original batch logic
        min_batch_size = BATCH_SETTINGS["min_batch_size"]
        if BATCH_SETTINGS["enabled"] and total_requests >= min_batch_size:
            processing_mode = "batch"
        else:
            processing_mode = "sequential"
    
    print(f"🔧 Processing mode: {processing_mode}")
    
    # Initialize RAG evaluators if mode is set
    # These will be passed to the chosen main evaluator
    langchain_rag_evaluator_instance = None
    graphrag_evaluator_instance = None

    if rag_mode == "langchain":
        # Consider passing pdfs_dir, index_dir from config if not using defaults in LangChainRAGEvaluator
        langchain_rag_evaluator_instance = LangChainRAGEvaluator()
        print(f"   Initialized LangChainRAGEvaluator for evaluation pipeline.")
    elif rag_mode == "graphrag":
        # Consider passing work_dir from config if not using defaults in GraphRAGEvaluator
        graphrag_evaluator_instance = GraphRAGEvaluator()
        print(f"   Initialized GraphRAGEvaluator for evaluation pipeline.")
        # For GraphRAG, its own setup (prepare_documents, run_indexing) should be managed
        # either externally (e.g., via CLI commands rag-graphrag-build) or ensure its init/retrieve handles it.

    # Run evaluation based on selected mode
    if processing_mode == "batch":
        print(f"🚀 Using Batch Processing")
        evaluator = OpenAIBatchEvaluator(
            model_name,
            langchain_rag_evaluator=langchain_rag_evaluator_instance,
            graphrag_evaluator=graphrag_evaluator_instance
        )
        results = evaluator.evaluate_dataset_batch(
            split=split,
            prompt_types=prompt_types,
            sample_size=sample_size,
            specialty_filter=specialty_filter,
            save_results=True,
            rag_mode=rag_mode,
            rag_params=rag_params
        )
    
    elif processing_mode == "concurrent":
        print(f"⚡ Using Concurrent Processing")
        max_concurrent_val = max_concurrent or CONCURRENT_CONFIG['max_concurrent_requests']
        requests_per_minute_val = requests_per_minute or CONCURRENT_CONFIG['requests_per_minute']
        
        evaluator = ConcurrentOpenAIEvaluator(
            model_name, 
            max_concurrent=max_concurrent_val,
            requests_per_minute=requests_per_minute_val,
            langchain_rag_evaluator=langchain_rag_evaluator_instance,
            graphrag_evaluator=graphrag_evaluator_instance
        )
        results = evaluator.evaluate_dataset_concurrent(
            split=split,
            prompt_types=prompt_types,
            sample_size=sample_size,
            specialty_filter=specialty_filter,
            save_results=True,
            rag_mode=rag_mode,
            rag_params=rag_params
        )
    
    else:  # sequential
        print(f"🔄 Using Sequential Processing")
        evaluator = OpenAIEvaluator(
            model_name,
            langchain_rag_evaluator=langchain_rag_evaluator_instance,
            graphrag_evaluator=graphrag_evaluator_instance
        )
        results = evaluator.evaluate_dataset(
            split=split,
            prompt_types=prompt_types,
            sample_size=sample_size,
            specialty_filter=specialty_filter,
            save_results=True,
            rag_mode=rag_mode,
            rag_params=rag_params
        )
    
    # Display results
    print("\n📈 Evaluation Results")
    print("-" * 40)
    
    summary = results['summary']
    overall = summary['overall_performance']
    
    print(f"Overall Accuracy: {overall['accuracy']:.3f}")
    print(f"Correct Answers: {overall['correct_count']}/{overall['total_count']}")
    if 'avg_response_time' in overall:
        print(f"Avg Response Time: {overall['avg_response_time']:.2f}s")
    
    # Performance by prompt type
    if len(prompt_types) > 1 or rag_mode:
        print(f"\n📝 Performance by Prompt Type (and RAG if applicable):")
        for prompt_type_key, perf in summary.get('performance_by_prompt', {}).items():
            # The prompt_type_key might now be e.g. "direct_rag"
            print(f"  {prompt_type_key}: {perf['accuracy']:.3f} ({perf['correct_count']}/{perf['total_count']})")
    
    # Performance by RAG mode
    if "performance_by_rag_mode" in summary:
        print(f"\n🧠 Performance by RAG Configuration:")
        for rag_config, perf in summary['performance_by_rag_mode'].items():
            if rag_config == "None": # Skip if RAG was not used for this subset
                continue
            print(f"  RAG Mode ('{rag_config}'): {perf['accuracy']:.3f} ({perf['correct_count']}/{perf['total_count']})")
    
    # Performance by specialty (top 5)
    if summary['performance_by_specialty']:
        print(f"\n🏥 Performance by Specialty (Top 5):")
        specialty_items = sorted(summary['performance_by_specialty'].items(), 
                               key=lambda x: x[1]['total_count'], reverse=True)
        for specialty, perf in specialty_items[:5]:
            print(f"  {specialty}: {perf['accuracy']:.3f} ({perf['correct_count']}/{perf['total_count']})")
    
    # API usage and performance metrics
    usage = results.get('api_usage', {})
    print(f"\n💰 API Usage & Performance:")
    print(f"  Processing mode: {processing_mode}")
    print(f"  Total API calls: {usage.get('total_calls', 0)}")
    if 'total_tokens' in usage:
        print(f"  Total tokens: {usage['total_tokens']:,}")
    if 'processing_time' in usage:
        print(f"  Processing time: {usage['processing_time']:.1f}s")
    if processing_mode == "concurrent" and 'requests_per_second' in usage:
        print(f"  Requests/second: {usage['requests_per_second']:.2f}")
    if usage.get('batch_processing'):
        print(f"  Cost savings: 50% (batch processing)")
    if usage.get('concurrent_processing'):
        print(f"  Speed boost: ~{usage.get('requests_per_second', 5):.0f}x vs sequential")
    
    return results

def download_best_practices():
    """Download BMJ Best Practice cardiology PDFs."""
    print("\n📚 Downloading BMJ Best Practice Cardiology Resources")
    print("-" * 60)
    
    downloader = BMJPDFDownloader()
    
    # Create topic list first
    print("Creating list of available topics...")
    downloader.create_topic_list()
    
    # Ask user if they want to proceed with downloads
    response = input("\nProceed with PDF downloads? This may take a while. (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # Download PDFs
    print("\nStarting PDF downloads...")
    summary = downloader.download_all_cardiology_pdfs(delay=2.0)
    
    print(f"\n✅ Download Summary:")
    print(f"  Total topics: {summary['total_topics']}")
    print(f"  Successful downloads: {summary['successful_count']}")
    print(f"  Failed downloads: {summary['failed_count']}")
    
    return summary

def interactive_mode():
    """Run in interactive mode for easy exploration."""
    print("\n🔍 Interactive Mode")
    print("Type 'help' for available commands, 'exit' to quit")
    
    loader = None
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'exit':
                break
            elif command == 'help':
                print("""
Available commands:
  analyze        - Analyze dataset statistics
  models         - List available models
  prompts        - List available prompt types
  specialties    - Show medical specialties in dataset
  eval           - Run quick evaluation (5 samples)
  full-eval      - Run full evaluation
  download       - Download BMJ Best Practice PDFs
  cardiology     - Evaluate on cardiology questions only
  filtered-6     - Evaluate on filtered test set (6 key specialties)
  batch-eval     - Run batch evaluation (concurrent processing)
  batch-full     - Run full batch evaluation
  batch-demo     - Demo batch processing (no waiting)
  batch-config   - Show batch processing configuration
  batch-status   - Check batch job status
  batch-list     - List recent batch jobs
  concurrent     - Run concurrent evaluation (faster than sequential)
  concurrent-config - Show concurrent processing configuration
  processing-modes - Compare different processing modes
  help           - Show this help
  exit           - Exit interactive mode
                """)
            
            elif command == 'analyze':
                loader = analyze_dataset()
            
            elif command == 'models':
                print("\nAvailable models:")
                for model in OPENAI_MODELS.keys():
                    print(f"  • {model}")
            
            elif command == 'prompts':
                print("\nAvailable prompt types:")
                for prompt_type in PromptTemplates.get_available_types():
                    print(f"  • {prompt_type}")
            
            elif command == 'specialties':
                if not loader:
                    loader = MedQADataLoader()
                specialties = loader.get_specialties('test')
                print("\nMedical specialties in test set:")
                for specialty, count in list(specialties.items())[:10]:
                    print(f"  • {specialty}: {count}")
            
            elif command == 'eval':
                run_evaluation(
                    model_name="gpt-3.5-turbo",
                    prompt_types=["direct"],
                    sample_size=5
                )
            
            elif command == 'full-eval':
                model = input("Model (gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"
                run_evaluation(
                    model_name=model,
                    prompt_types=["direct", "chain_of_thought"]
                )
            
            elif command == 'cardiology':
                run_evaluation(
                    model_name="gpt-3.5-turbo",
                    prompt_types=["direct", "chain_of_thought"],
                    specialty_filter="Cardiology"
                )
            
            elif command == 'download':
                download_best_practices()
            
            elif command == 'filtered-6':
                run_evaluation(
                    model_name="gpt-3.5-turbo",
                    split="test_filtered_6",
                    prompt_types=["direct", "chain_of_thought"]
                )
            
            elif command == 'batch-eval':
                model = input("Model (gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"
                run_evaluation(
                    model_name=model,
                    prompt_types=["direct", "chain_of_thought"],
                    use_batch=True
                )
            
            elif command == 'batch-full':
                model = input("Model (gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"
                run_evaluation(
                    model_name=model,
                    prompt_types=["direct", "chain_of_thought"],
                    use_batch=True,
                    sample_size=None
                )
            
            elif command == 'batch-demo':
                model = input("Model (gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"
                
                # Temporarily enable demo mode
                original_demo = BATCH_SETTINGS["demo_mode"]
                BATCH_SETTINGS["demo_mode"] = True
                
                try:
                    run_evaluation(
                        model_name=model,
                        prompt_types=["direct", "chain_of_thought"],
                        use_batch=True,
                        sample_size=20  # Reasonable demo size
                    )
                finally:
                    # Restore original demo mode setting
                    BATCH_SETTINGS["demo_mode"] = original_demo
            
            elif command == 'batch-config':
                print("\nBatch Processing Configuration:")
                print(f"  Enabled: {BATCH_SETTINGS['enabled']}")
                print(f"  Min batch size: {BATCH_SETTINGS['min_batch_size']}")
                print(f"  Max batch size: {BATCH_SETTINGS['max_batch_size']}")
                print(f"  Poll interval: {BATCH_SETTINGS['poll_interval']}s")
                print(f"  Demo mode: {BATCH_SETTINGS['demo_mode']}")
                print(f"  Auto cleanup: {BATCH_SETTINGS['auto_cleanup']}")
            
            elif command == 'batch-status':
                batch_id = input("Batch ID: ").strip()
                if batch_id:
                    check_batch_status(batch_id)
                else:
                    print("No batch ID provided")
            
            elif command == 'batch-list':
                list_batch_jobs()
            
            elif command == 'concurrent':
                print("\n⚡ Concurrent Evaluation")
                print("=" * 40)
                
                # Get parameters
                model = input("Model (gpt-4o-mini): ").strip() or "gpt-4o-mini"
                split = input("Split (test_filtered_6): ").strip() or "test_filtered_6"
                sample_size = input("Sample size (10): ").strip()
                sample_size = int(sample_size) if sample_size else 10
                
                max_concurrent = input(f"Max concurrent requests ({CONCURRENT_CONFIG['max_concurrent_requests']}): ").strip()
                max_concurrent = int(max_concurrent) if max_concurrent else CONCURRENT_CONFIG['max_concurrent_requests']
                
                requests_per_minute = input(f"Requests per minute ({CONCURRENT_CONFIG['requests_per_minute']}): ").strip()
                requests_per_minute = int(requests_per_minute) if requests_per_minute else CONCURRENT_CONFIG['requests_per_minute']
                
                reasoning_types = input("Reasoning types (direct,chain_of_thought): ").strip()
                reasoning_types = reasoning_types.split(',') if reasoning_types else ['direct', 'chain_of_thought']
                reasoning_types = [r.strip() for r in reasoning_types]
                
                # Run concurrent evaluation
                evaluator = ConcurrentOpenAIEvaluator(
                    model, 
                    max_concurrent=max_concurrent, 
                    requests_per_minute=requests_per_minute
                )
                
                results = evaluator.evaluate_dataset_concurrent(
                    split=split,
                    prompt_types=reasoning_types,
                    sample_size=sample_size,
                    save_results=True
                )
                
                if results and 'summary' in results:
                    summary = results['summary']
                    usage = results['api_usage']
                    print(f"\n✅ Concurrent evaluation completed!")
                    print(f"📊 Overall accuracy: {summary['overall_performance']['accuracy']:.3f}")
                    print(f"⚡ Requests/second: {usage['requests_per_second']:.2f}")
                    print(f"💰 Total API calls: {usage['total_calls']}")
                    print(f"🕒 Processing time: {usage['processing_time']:.1f} seconds")
            
            elif command == 'concurrent-config':
                print("\n⚡ Concurrent Processing Configuration")
                print("=" * 45)
                print(f"Max concurrent requests: {CONCURRENT_CONFIG['max_concurrent_requests']}")
                print(f"Requests per minute: {CONCURRENT_CONFIG['requests_per_minute']}")
                print(f"Concurrent enabled: {CONCURRENT_CONFIG['enable_concurrent']}")
                print(f"Concurrent threshold: {CONCURRENT_CONFIG['concurrent_threshold']}")
                print(f"\nEvaluation Settings:")
                print(f"Auto concurrent: {EVALUATION_SETTINGS['auto_concurrent']}")
                print(f"Concurrent threshold: {EVALUATION_SETTINGS['concurrent_threshold']}")
                print(f"Batch threshold: {EVALUATION_SETTINGS['batch_threshold']}")
            
            elif command == 'processing-modes':
                print("\n🔧 Processing Mode Comparison")
                print("=" * 40)
                print("Sequential:")
                print("  ✅ Simple and reliable")
                print("  ❌ Slow for large evaluations")
                print("  📊 ~1-2 requests/second")
                print()
                print("Concurrent:")
                print("  ✅ Much faster than sequential")
                print("  ✅ Same cost as sequential")
                print("  ✅ Respects rate limits")
                print("  📊 ~5-20 requests/second")
                print("  ⚠️  More complex error handling")
                print()
                print("Batch (OpenAI):")
                print("  ✅ 50% cheaper than other modes")
                print("  ❌ Very slow (10min - 24hrs)")
                print("  ✅ Perfect for large production runs")
                print("  📊 Unlimited throughput but delayed")
                print()
                print("Recommendations:")
                print("  • <5 requests: Sequential")
                print("  • 5-100 requests: Concurrent")
                print("  • 100+ requests: Consider Batch for cost savings")
            
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit.")
        except Exception as e:
            print(f"Error: {e}")

def run_rag_langchain_build():
    """Build LangChain RAG index from cardiology PDFs."""
    print("\n🧱 Building LangChain RAG Index...")
    print("-" * 40)
    
    rag = LangChainRAGEvaluator()
    chunks_count = rag.load_and_index_documents()
    
    print(f"\n✅ LangChain RAG index built successfully!")
    print(f"   Total chunks indexed: {chunks_count}")
    stats = rag.get_retrieval_stats()
    print(f"   Vector store stats: {stats}")

def run_rag_langchain_eval(sample_size: Optional[int] = 5):
    """Evaluate LangChain RAG on sample cardiology questions."""
    print("\n🧪 Evaluating LangChain RAG...")
    print("-" * 40)
    
    rag = LangChainRAGEvaluator()
    if not rag.load_existing_vectorstore():
        print("❌ No existing vector store found. Please build the index first using 'rag-langchain-build'.")
        return
        
    # Sample questions (can be expanded or loaded from a file)
    test_questions = [
        "What are the main symptoms of myocardial infarction?",
        "How is atrial fibrillation diagnosed and treated?",
        "What are the risk factors for coronary artery disease?",
        "Explain the pathophysiology of heart failure with reduced ejection fraction.",
        "What are the contraindications for beta-blockers in cardiology?"
    ][:sample_size]
    
    results = rag.evaluate_on_dataset(test_questions)
    
    # Print summary of results
    print("\n📈 LangChain RAG Evaluation Summary:")
    for res in results:
        print(f"  ❓ Question: {res['question']}")
        print(f"  💡 Answer: {res['answer'][:150]}...")
        print(f"  📚 Sources: {len(res['sources'])} documents")
        print(f"  ⏱️ Time: {res['query_time']:.2f}s\n")

def run_rag_graphrag_build():
    """Build GraphRAG index from cardiology PDFs."""
    print("\n🕸️ Building GraphRAG Index...")
    print("-" * 40)
    
    graphrag = GraphRAGEvaluator()
    
    # Prepare documents (convert PDFs to text)
    doc_count = graphrag.prepare_documents()
    if doc_count == 0:
        print("❌ No documents prepared for GraphRAG. Check 'cardiology_pdfs' directory.")
        return
        
    # Create settings.yaml
    graphrag.create_settings_yaml()
    
    # Run indexing
    if not graphrag.run_indexing():
        print("❌ GraphRAG indexing failed.")
        return
        
    print("\n✅ GraphRAG index built successfully!")
    stats = graphrag.get_index_stats()
    print(f"   Index stats: {stats}")

def run_rag_graphrag_eval(method: str = "global", sample_size: Optional[int] = 5):
    """Run a quick GraphRAG evaluation test."""
    print(f"\n🧪 Running GraphRAG Evaluation Test ({method} method)")
    print("-" * 50)
    
    try:
        # Initialize GraphRAG evaluator
        evaluator = GraphRAGEvaluator()
        
        # Test query
        test_question = "What are the main treatments for heart failure with reduced ejection fraction?"
        print(f"Test question: {test_question}")
        
        # Query GraphRAG
        result = evaluator.query_global(test_question) if method == "global" else evaluator.query_local(test_question)
        print(f"\nGraphRAG Response ({method}):")
        print(result)
        
        # Run evaluation on small sample
        print(f"\n📊 Running evaluation on {sample_size} samples...")
        from model_evaluator import OpenAIEvaluator
        
        eval_instance = OpenAIEvaluator("gpt-4o-mini", graphrag_evaluator=evaluator)
        results = eval_instance.evaluate_dataset(
            split="test",
            prompt_types=["direct"],
            sample_size=sample_size,
            specialty_filter="Cardiology",
            save_results=True,
            rag_mode="graphrag",
            rag_params={"graphrag_search_type": method, "k_retrieval": 3}
        )
        
        print(f"✅ GraphRAG evaluation completed!")
        print(f"Accuracy: {results['summary']['overall_performance']['accuracy']:.3f}")
        
    except Exception as e:
        print(f"❌ Error in GraphRAG evaluation: {e}")
        print("Make sure GraphRAG workspace is set up and indexed.")

def run_adaptive_rag_eval(k_guidelines: int = 3, 
                         k_retrieval_per_guideline: int = 2,
                         sample_size: Optional[int] = 3,
                         prompt_types: List[str] = ["direct"]):
    """Run adaptive RAG evaluation test."""
    print(f"\n🧪 Running Adaptive RAG Evaluation Test")
    print("-" * 50)
    print(f"Guidelines per question: {k_guidelines}")
    print(f"Retrievals per guideline: {k_retrieval_per_guideline}")
    print(f"Sample size: {sample_size}")
    print(f"Prompt types: {prompt_types}")
    
    try:
        # Initialize adaptive RAG evaluator
        evaluator = AdaptiveRAGEvaluator(
            model_name="gpt-4o-mini",
            k_guidelines=k_guidelines,
            k_retrieval_per_guideline=k_retrieval_per_guideline
        )
        
        # Run evaluation
        results = evaluator.evaluate_dataset_adaptive(
            split="test",
            prompt_types=prompt_types,
            sample_size=sample_size,
            specialty_filter="Cardiology",
            save_results=True
        )
        
        print(f"\n✅ Adaptive RAG evaluation completed!")
        print(f"Overall accuracy: {results['summary']['overall_performance']['accuracy']:.3f}")
        print(f"Average time per sample: {results['summary']['adaptive_rag_metrics']['avg_total_time']:.1f}s")
        print(f"Average API calls per sample: {results['summary']['adaptive_rag_metrics']['avg_api_calls_per_sample']:.1f}")
        
        # Show stage breakdown
        metrics = results['summary']['adaptive_rag_metrics']
        print(f"\n⏱️  Stage Timing Breakdown:")
        print(f"  Stage 1 (Guideline ID): {metrics['avg_stage1_time']:.2f}s")
        print(f"  Stage 2 (Retrieval): {metrics['avg_stage2_time']:.2f}s") 
        print(f"  Stage 3 (Final Answer): {metrics['avg_stage3_time']:.2f}s")
        
    except Exception as e:
        print(f"❌ Error in adaptive RAG evaluation: {e}")
        print("Make sure LangChain RAG index is built and available.")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Medical Reasoning Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --analyze                    # Analyze dataset
  python main.py --eval --model gpt-4         # Quick evaluation with GPT-4
  python main.py --eval --full                # Full evaluation
  python main.py --eval --cardiology          # Evaluate cardiology questions
  python main.py --download                   # Download BMJ PDFs
  python main.py --interactive                # Interactive mode
  
  # RAG Examples:
  python main.py --rag-langchain-build        # Build LangChain RAG index
  python main.py --rag-langchain-eval 10      # Test LangChain RAG on 10 samples
  python main.py --adaptive-rag-eval 3        # Run adaptive RAG with 3 guidelines
  python main.py --eval --rag-mode langchain  # Evaluate with LangChain RAG
        """
    )
    
    # Main action arguments
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset statistics')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    parser.add_argument('--download', action='store_true', help='Download BMJ Best Practice PDFs')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    # Evaluation configuration
    parser.add_argument('--model', default='gpt-3.5-turbo', 
                       choices=list(OPENAI_MODELS.keys()),
                       help='OpenAI model to use')
    parser.add_argument('--split', default='test', 
                       choices=['train', 'validation', 'test', 'test_filtered_6'],
                       help='Dataset split to use')
    parser.add_argument('--prompts', nargs='+', 
                       default=['direct'],
                       choices=PromptTemplates.get_available_types(),
                       help='Prompt types to use')
    parser.add_argument('--sample-size', type=int, default=5,
                       help='Number of samples to evaluate (default: 5 for quick test)')
    parser.add_argument('--specialty', type=str,
                       help='Filter by medical specialty')
    parser.add_argument('--full', action='store_true',
                       help='Run full evaluation (overrides sample-size)')
    parser.add_argument('--cardiology', action='store_true',
                       help='Evaluate cardiology questions only')
    parser.add_argument('--filtered-6', action='store_true',
                       help='Use filtered test set with 6 key specialties')
    parser.add_argument('--batch', action='store_true',
                       help='Force use of batch processing')
    parser.add_argument('--no-batch', action='store_true',
                       help='Force use of synchronous processing')
    parser.add_argument('--batch-demo', action='store_true',
                       help='Demo batch processing (simulated, no waiting)')
    
    # Add concurrent processing arguments
    parser.add_argument("--concurrent", action="store_true", 
                       help="Use concurrent processing for faster evaluation")
    parser.add_argument("--max-concurrent", type=int, default=CONCURRENT_CONFIG['max_concurrent_requests'],
                       help="Maximum concurrent requests (default: 10)")
    parser.add_argument("--requests-per-minute", type=int, default=CONCURRENT_CONFIG['requests_per_minute'],
                       help="Rate limit for requests per minute (default: 100)")
    
    # Add RAG arguments
    parser.add_argument("--rag-langchain-build", action="store_true", help="Build LangChain RAG index")
    parser.add_argument("--rag-langchain-eval", nargs='?', const=5, type=int, metavar='N', help="Evaluate LangChain RAG (optional N samples)")
    parser.add_argument("--rag-graphrag-build", action="store_true", help="Build GraphRAG index")
    parser.add_argument("--rag-graphrag-eval-global", nargs='?', const=5, type=int, metavar='N', help="Evaluate GraphRAG with global search (optional N samples)")
    parser.add_argument("--rag-graphrag-eval-local", nargs='?', const=5, type=int, metavar='N', help="Evaluate GraphRAG with local search (optional N samples)")
    parser.add_argument("--adaptive-rag-eval", nargs='?', const=3, type=int, metavar='K', help="Run adaptive RAG evaluation (optional K guidelines, default 3)")
    
    # Add RAG related arguments
    rag_group = parser.add_argument_group('RAG Settings', "Options for Retrieval Augmented Generation")
    rag_group.add_argument("--rag-mode", type=str, choices=["langchain", "graphrag"], default=None,
                         help="Enable RAG mode. Choose between 'langchain' or 'graphrag'.")
    rag_group.add_argument("--rag-k-retrieval", type=int, default=5,
                         help="Number of documents/context pieces to retrieve for RAG.")
    rag_group.add_argument("--graphrag-search-type", type=str, choices=["global", "local"], default="global",
                         help="Search type for GraphRAG ('global' or 'local').")
    rag_group.add_argument("--graphrag-community-level", type=int, default=2,
                         help="Community level for GraphRAG local search.")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Handle different modes
    if args.interactive:
        interactive_mode()
    
    elif args.analyze:
        analyze_dataset()
    
    elif args.eval:
        sample_size = None if args.full else args.sample_size
        specialty_filter = "Cardiology" if args.cardiology else args.specialty
        split = "test_filtered_6" if args.filtered_6 else args.split
        
        # Determine processing mode preferences
        use_batch = None
        use_concurrent = None
        
        if args.no_batch:
            use_batch = False
        elif args.batch or args.batch_demo:
            use_batch = True
        
        if args.concurrent:
            use_concurrent = True
        
        # Handle demo mode
        if args.batch_demo:
            original_demo = BATCH_SETTINGS["demo_mode"]
            BATCH_SETTINGS["demo_mode"] = True
        
        # Collect RAG parameters
        rag_params_dict = None
        if args.rag_mode:
            rag_params_dict = {
                "k_retrieval": args.rag_k_retrieval,
                "graphrag_search_type": args.graphrag_search_type,
                "graphrag_community_level": args.graphrag_community_level
            }
        
        try:
            run_evaluation(
                model_name=args.model,
                split=split,
                prompt_types=args.prompts,
                sample_size=sample_size,
                specialty_filter=specialty_filter,
                use_batch=use_batch,
                use_concurrent=use_concurrent,
                max_concurrent=args.max_concurrent,
                requests_per_minute=args.requests_per_minute,
                rag_mode=args.rag_mode,
                rag_params=rag_params_dict
            )
        finally:
            # Restore demo mode if it was changed
            if args.batch_demo:
                BATCH_SETTINGS["demo_mode"] = original_demo
    
    elif args.download:
        download_best_practices()
    
    elif args.rag_langchain_build:
        run_rag_langchain_build()
    elif args.rag_langchain_eval is not None:
        run_rag_langchain_eval(sample_size=args.rag_langchain_eval)
    elif args.rag_graphrag_build:
        run_rag_graphrag_build()
    elif args.rag_graphrag_eval_global is not None:
        run_rag_graphrag_eval(method="global", sample_size=args.rag_graphrag_eval_global)
    elif args.rag_graphrag_eval_local is not None:
        run_rag_graphrag_eval(method="local", sample_size=args.rag_graphrag_eval_local)
    
    elif args.adaptive_rag_eval is not None:
        run_adaptive_rag_eval(k_guidelines=args.adaptive_rag_eval)
    
    else:
        print("No action specified. Use --help for usage information.")
        print("Tip: Try 'python main.py --interactive' for interactive mode.")

if __name__ == "__main__":
    main() 