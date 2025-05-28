"""Model evaluation system for medical reasoning tasks."""
import openai
import json
import time
import re
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from config import OPENAI_API_KEY, OPENAI_MODELS
from data_loader import MedQADataLoader
from reasoning_prompts import PromptTemplates

# Potentially import RAG evaluators if they are to be instantiated here
# from rag_langchain import LangChainRAGEvaluator
# from rag_graphrag import GraphRAGEvaluator
# For now, we\'ll assume they are passed in or handled by the caller of evaluate_sample

class OpenAIEvaluator:
    """Evaluator for OpenAI models on medical reasoning tasks."""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 langchain_rag_evaluator: Optional[object] = None, # Allow passing pre-initialized RAG evaluators
                 graphrag_evaluator: Optional[object] = None):
        """Initialize the evaluator with a specific model and optional RAG evaluators."""
        if model_name not in OPENAI_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(OPENAI_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_config = OPENAI_MODELS[model_name]
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize data loader
        self.data_loader = MedQADataLoader()

        # Store RAG evaluators
        self.langchain_rag_evaluator = langchain_rag_evaluator
        self.graphrag_evaluator = graphrag_evaluator
        
        # Track API usage
        self.api_calls = 0
        self.total_tokens = 0
    
    def _make_api_call(self, prompt: str, max_retries: int = 3) -> str:
        """Make an API call to OpenAI with retries."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.model_config["max_tokens"],
                    temperature=self.model_config["temperature"],
                    top_p=self.model_config["top_p"],
                    frequency_penalty=self.model_config.get("frequency_penalty", 0.0),
                    presence_penalty=self.model_config.get("presence_penalty", 0.0),
                    seed=self.model_config.get("seed", 42)
                )
                
                self.api_calls += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens += response.usage.total_tokens
                
                return response.choices[0].message.content.strip()
                
            except openai.RateLimitError:
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            except openai.APIError as e:
                print(f"API error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
            except Exception as e:
                print(f"Unexpected error: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
        
        raise Exception(f"Failed to get response after {max_retries} attempts")
    
    def _extract_answer(self, response: str) -> str:
        """Extract the answer choice from the <answer> tag in model response."""
        response = response.strip()
        
        # Primary method: Look for <answer>X</answer> tag
        answer_match = re.search(r'<answer>\s*([A-Z])\s*</answer>', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Fallback: Look for answer tag without closing tag
        answer_match = re.search(r'<answer>\s*([A-Z])', response, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
        
        # Secondary fallback: Legacy patterns for responses that don't use tags
        legacy_patterns = [
            r'(?:Therefore|Thus|Hence),?\s+(?:the\s+)?correct\s+answer\s+is\s+([A-Z])\.',  # "Therefore, the correct answer is A."
            r'(?:Answer|answer):\s*([A-Z])\b',  # "Answer: A"
            r'(?:Thus|Therefore|Hence),?\s*(?:the\s+)?(?:answer\s+is\s+)?\*?\*?([A-Z])\b',  # "Thus, the answer is **A**"
        ]
        
        for pattern in legacy_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Last resort: look for the last capital letter in valid range
        letters = re.findall(r'\b([A-E])\b', response)
        if letters:
            return letters[-1].upper()
        
        return "UNKNOWN"
    
    def evaluate_sample(self, 
                        sample: Dict, 
                        prompt_type: str = "direct",
                        rag_mode: Optional[str] = None, # e.g., "langchain", "graphrag"
                        rag_params: Optional[Dict] = None) -> Dict: # For specific RAG params like k, search_type
        """Evaluate a single sample, with optional RAG augmentation."""
        question = sample['Question']
        options = sample['Options']
        correct_answer = self.data_loader.get_correct_answer(sample)
        correct_choice = self.data_loader.get_answer_choice(sample)
        
        context: Optional[str] = None
        rag_source_info: Optional[Dict] = None
        actual_prompt_type = prompt_type

        if rag_mode:
            rag_params = rag_params or {}
            print(f"  Enhancing with {rag_mode} RAG (params: {rag_params})...")
            retrieved_context_data = None
            
            try:
                if rag_mode == "langchain":
                    if not self.langchain_rag_evaluator:
                        # Dynamically import and initialize if not provided
                        from rag_langchain import LangChainRAGEvaluator
                        # Use default paths or allow configuration
                        self.langchain_rag_evaluator = LangChainRAGEvaluator() 
                        print("    Initialized LangChainRAGEvaluator on-demand.")
                    
                    retrieved_context_data = self.langchain_rag_evaluator.retrieve_context(
                        question, 
                        k=rag_params.get('k_retrieval', self.langchain_rag_evaluator.k_retrieval if hasattr(self.langchain_rag_evaluator, 'k_retrieval') else 5)
                    )
                    context = retrieved_context_data # retrieve_context returns the formatted string
                    rag_source_info = {"type": "langchain", "retrieved_text": context} # Simplified source info

                elif rag_mode == "graphrag":
                    if not self.graphrag_evaluator:
                        from rag_graphrag import GraphRAGEvaluator # Requires async handling
                        self.graphrag_evaluator = GraphRAGEvaluator()
                        print("    Initialized GraphRAGEvaluator on-demand.")

                    # GraphRAG methods are async, need to handle this
                    # This is a blocking call for simplicity in this synchronous evaluator.
                    # For concurrent/batch, this will need proper async integration.
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError: # 'no current event loop in thread'
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    retrieved_context_data = loop.run_until_complete(
                        self.graphrag_evaluator.retrieve_context(
                            question,
                            search_type=rag_params.get('graphrag_search_type', "global"),
                            k=rag_params.get('k_retrieval', 5),
                            community_level=rag_params.get('graphrag_community_level', 2)
                        )
                    )
                    context = retrieved_context_data # retrieve_context returns the formatted string
                    rag_source_info = {"type": "graphrag", "search_type": rag_params.get('graphrag_search_type', "global"), "retrieved_text": context}
                
                else:
                    print(f"    Warning: Unknown RAG mode '{rag_mode}'. Proceeding without RAG.")

                if context and "_rag" not in prompt_type: # Ensure we use a RAG variant of the prompt
                    actual_prompt_type = f"{prompt_type}_rag"
                elif not context:
                    print(f"    Warning: RAG mode '{rag_mode}' was specified but no context was retrieved for the question. Falling back to non-RAG prompt.")
                    actual_prompt_type = prompt_type # Fallback to non-RAG if context is empty
                    rag_mode = None # Nullify RAG mode if no context

            except Exception as e:
                print(f"    Error during RAG context retrieval ({rag_mode}): {e}. Proceeding without RAG.")
                context = None
                rag_source_info = {"type": rag_mode, "error": str(e)}
                actual_prompt_type = prompt_type # Fallback to non-RAG prompt
                rag_mode = None # Nullify RAG mode due to error


        # Generate prompt
        try:
            prompt = PromptTemplates.get_prompt(actual_prompt_type, question, options, context=context)
        except ValueError as e:
            print(f"Error getting prompt for type '{actual_prompt_type}' (context provided: {context is not None}): {e}")
            print(f"Falling back to non-RAG prompt type '{prompt_type}'.")
            actual_prompt_type = prompt_type # Fallback
            prompt = PromptTemplates.get_prompt(actual_prompt_type, question, options) # Context will be ignored by non-RAG prompt
        
        # Get model response
        start_time = time.time()
        response = self._make_api_call(prompt)
        response_time = time.time() - start_time
        
        # Extract predicted answer
        predicted_choice = self._extract_answer(response)
        is_correct = predicted_choice == correct_choice
        
        result = {
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'correct_choice': correct_choice,
            'prompt_type': actual_prompt_type, # Log the actual prompt type used
            'full_prompt': prompt,  # Save the complete prompt sent to LLM
            'model_response': response,
            'predicted_choice': predicted_choice,
            'is_correct': is_correct,
            'response_time': response_time,
            'specialty': self.data_loader.get_sample_specialty(sample),
            'rag_mode': rag_mode, # Log if RAG was used
            'rag_source_info': rag_source_info # Log context source or error
        }
        return result
    
    def evaluate_dataset(self, 
                        split: str = "test", 
                        prompt_types: List[str] = ["direct"],
                        sample_size: Optional[int] = None,
                        specialty_filter: Optional[str] = None,
                        save_results: bool = True,
                        rag_mode: Optional[str] = None, # Added RAG mode
                        rag_params: Optional[Dict] = None) -> Dict: # Added RAG params
        """Evaluate the model on a dataset split, with optional RAG."""
        
        print(f"Starting evaluation on {split} split")
        print(f"Model: {self.model_name}")
        print(f"Prompt types: {prompt_types}")
        if rag_mode:
            print(f"RAG Mode: {rag_mode}")
            if rag_params:
                print(f"RAG Params: {rag_params}")

        # Initialize RAG evaluators if mode is set and they are not pre-initialized
        # This is a basic way; a more robust setup might involve a factory or dependency injection
        if rag_mode == "langchain" and not self.langchain_rag_evaluator:
            from rag_langchain import LangChainRAGEvaluator
            self.langchain_rag_evaluator = LangChainRAGEvaluator() # Consider passing pdfs_dir, index_dir from config
            print(f"Initialized LangChainRAGEvaluator for dataset evaluation.")
        
        if rag_mode == "graphrag" and not self.graphrag_evaluator:
            from rag_graphrag import GraphRAGEvaluator
            self.graphrag_evaluator = GraphRAGEvaluator() # Consider passing work_dir from config
            print(f"Initialized GraphRAGEvaluator for dataset evaluation.")
            # For GraphRAG, ensure its setup (like prepare_documents, run_indexing) is called
            # This might be done here or expected to be done externally before evaluation.
            # For simplicity, we assume it's ready or `retrieve_context` handles it.

        # Get data
        data = self.data_loader.get_split(split, sample_size, specialty_filter)
        # Sort data by question text to ensure consistent processing order
        data = sorted(data, key=lambda x: x['Question'])
        print(f"Evaluating on {len(data)} samples")
        
        all_results = []
        
        for prompt_type in prompt_types:
            print(f"\nEvaluating with {prompt_type}{f' +{rag_mode}' if rag_mode else ''} prompts...")
            
            prompt_results = []
            for sample in tqdm(data, desc=f"Processing {prompt_type}{f'+{rag_mode}' if rag_mode else ''}"):
                try:
                    result = self.evaluate_sample(sample, prompt_type, rag_mode=rag_mode, rag_params=rag_params)
                    prompt_results.append(result)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue
            
            # Calculate accuracy for this prompt type
            correct_count = sum(1 for r in prompt_results if r['is_correct'])
            accuracy = correct_count / len(prompt_results) if prompt_results else 0
            print(f"{prompt_type}{f'+{rag_mode}' if rag_mode else ''} accuracy: {accuracy:.3f} ({correct_count}/{len(prompt_results)})")
        
        # Generate comprehensive results
        results_summary = self._generate_results_summary(all_results, split)
        
        if save_results:
            self._save_results(all_results, results_summary, split)
        
        return {
            'detailed_results': all_results,
            'summary': results_summary,
            'api_usage': {
                'total_calls': self.api_calls,
                'total_tokens': self.total_tokens
            }
        }
    
    def _generate_results_summary(self, results: List[Dict], split: str) -> Dict:
        """Generate comprehensive results summary."""
        df = pd.DataFrame(results)
        
        summary = {
            'evaluation_info': {
                'model': self.model_name,
                'split': split,
                'total_samples': len(results),
                'timestamp': datetime.now().isoformat()
            },
            'overall_performance': {},
            'performance_by_prompt': {},
            'performance_by_specialty': {},
            'error_analysis': {}
        }
        
        # Overall performance
        overall_accuracy = float(df['is_correct'].mean())
        summary['overall_performance']['accuracy'] = overall_accuracy
        summary['overall_performance']['correct_count'] = int(df['is_correct'].sum())
        summary['overall_performance']['total_count'] = len(df)
        summary['overall_performance']['avg_response_time'] = float(df['response_time'].mean())
        
        # Performance by prompt type
        for prompt_type in df['prompt_type'].unique():
            prompt_df = df[df['prompt_type'] == prompt_type]
            summary['performance_by_prompt'][prompt_type] = {
                'accuracy': float(prompt_df['is_correct'].mean()),
                'correct_count': int(prompt_df['is_correct'].sum()),
                'total_count': len(prompt_df),
                'avg_response_time': float(prompt_df['response_time'].mean())
            }
        
        # Performance by specialty
        for specialty in df['specialty'].unique():
            specialty_df = df[df['specialty'] == specialty]
            summary['performance_by_specialty'][specialty] = {
                'accuracy': float(specialty_df['is_correct'].mean()),
                'correct_count': int(specialty_df['is_correct'].sum()),
                'total_count': len(specialty_df)
            }
        
        # Performance by RAG mode
        if 'rag_mode' in df.columns:
            summary['performance_by_rag_mode'] = {}
            for mode in df['rag_mode'].fillna('None').unique():
                mode_df = df[df['rag_mode'].fillna('None') == mode]
                summary['performance_by_rag_mode'][mode] = {
                    'accuracy': float(mode_df['is_correct'].mean()),
                    'correct_count': int(mode_df['is_correct'].sum()),
                    'total_count': len(mode_df),
                    'avg_response_time': float(mode_df['response_time'].mean() if not mode_df.empty else 0)
                }

        # Error analysis
        incorrect_df = df[~df['is_correct']]
        summary['error_analysis']['total_errors'] = len(incorrect_df)
        summary['error_analysis']['unknown_answers'] = int((df['predicted_choice'] == 'UNKNOWN').sum())
        
        if len(incorrect_df) > 0:
            # Convert value_counts to regular dict with int values
            errors_by_specialty = incorrect_df['specialty'].value_counts().to_dict()
            errors_by_prompt = incorrect_df['prompt_type'].value_counts().to_dict()
            
            summary['error_analysis']['errors_by_specialty'] = {k: int(v) for k, v in errors_by_specialty.items()}
            summary['error_analysis']['errors_by_prompt'] = {k: int(v) for k, v in errors_by_prompt.items()}
        
        # Error analysis by RAG mode
        if 'rag_mode' in incorrect_df.columns:
            summary['error_analysis']['errors_by_rag_mode'] = {}
            errors_by_rag = incorrect_df['rag_mode'].fillna('None').value_counts().to_dict()
            summary['error_analysis']['errors_by_rag_mode'] = {k: int(v) for k,v in errors_by_rag.items()}
        
        return summary
    
    def _save_results(self, results: List[Dict], summary: Dict, split: str) -> None:
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs("results", exist_ok=True)
        
        # Save detailed results
        detailed_file = f"results/{self.model_name}_{split}_{timestamp}_detailed.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_file = f"results/{self.model_name}_{split}_{timestamp}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = f"results/{self.model_name}_{split}_{timestamp}.csv"
        pd.DataFrame(results).to_csv(csv_file, index=False)
        
        print(f"\nResults saved:")
        print(f"  Detailed: {detailed_file}")
        print(f"  Summary: {summary_file}")
        print(f"  CSV: {csv_file}")

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = OpenAIEvaluator("gpt-3.5-turbo")
    
    # Quick test on a small sample
    print("Running quick test evaluation...")
    results = evaluator.evaluate_dataset(
        split="test",
        prompt_types=["direct", "chain_of_thought"],
        sample_size=5,  # Small sample for testing
        save_results=True
    )
    
    print(f"\nQuick test completed!")
    print(f"Overall accuracy: {results['summary']['overall_performance']['accuracy']:.3f}")
    print(f"API calls made: {results['api_usage']['total_calls']}")
    print(f"Total tokens used: {results['api_usage']['total_tokens']}") 