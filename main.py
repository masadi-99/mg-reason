#!/usr/bin/env python3
"""Main script for medical reasoning evaluation system."""
import argparse
import json
import os
from typing import List, Optional

from config import OPENAI_MODELS, EVALUATION_SETTINGS
from data_loader import MedQADataLoader
from model_evaluator import OpenAIEvaluator
from reasoning_prompts import PromptTemplates
from bmj_pdf_downloader import BMJPDFDownloader

def print_banner():
    """Print a welcome banner."""
    print("=" * 60)
    print("🏥 Medical Reasoning Evaluation System")
    print("   Exploring AI Performance on Medical Datasets")
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
                  specialty_filter: Optional[str] = None):
    """Run evaluation on specified configuration."""
    
    print(f"\n🤖 Starting Evaluation")
    print("-" * 40)
    print(f"Model: {model_name}")
    print(f"Split: {split}")
    print(f"Prompt types: {', '.join(prompt_types)}")
    if sample_size:
        print(f"Sample size: {sample_size}")
    if specialty_filter:
        print(f"Specialty filter: {specialty_filter}")
    
    # Initialize evaluator
    evaluator = OpenAIEvaluator(model_name)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(
        split=split,
        prompt_types=prompt_types,
        sample_size=sample_size,
        specialty_filter=specialty_filter,
        save_results=True
    )
    
    # Display results
    print("\n📈 Evaluation Results")
    print("-" * 40)
    
    summary = results['summary']
    overall = summary['overall_performance']
    
    print(f"Overall Accuracy: {overall['accuracy']:.3f}")
    print(f"Correct Answers: {overall['correct_count']}/{overall['total_count']}")
    print(f"Avg Response Time: {overall['avg_response_time']:.2f}s")
    
    # Performance by prompt type
    if len(prompt_types) > 1:
        print(f"\n📝 Performance by Prompt Type:")
        for prompt_type, perf in summary['performance_by_prompt'].items():
            print(f"  {prompt_type}: {perf['accuracy']:.3f} ({perf['correct_count']}/{perf['total_count']})")
    
    # Performance by specialty (top 5)
    print(f"\n🏥 Performance by Specialty (Top 5):")
    specialty_items = sorted(summary['performance_by_specialty'].items(), 
                           key=lambda x: x[1]['total_count'], reverse=True)
    for specialty, perf in specialty_items[:5]:
        print(f"  {specialty}: {perf['accuracy']:.3f} ({perf['correct_count']}/{perf['total_count']})")
    
    # API usage
    usage = results['api_usage']
    print(f"\n💰 API Usage:")
    print(f"  Total API calls: {usage['total_calls']}")
    print(f"  Total tokens: {usage['total_tokens']:,}")
    
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
            
            else:
                print(f"Unknown command: {command}. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit.")
        except Exception as e:
            print(f"Error: {e}")

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
        
        run_evaluation(
            model_name=args.model,
            split=split,
            prompt_types=args.prompts,
            sample_size=sample_size,
            specialty_filter=specialty_filter
        )
    
    elif args.download:
        download_best_practices()
    
    else:
        print("No action specified. Use --help for usage information.")
        print("Tip: Try 'python main.py --interactive' for interactive mode.")

if __name__ == "__main__":
    main() 