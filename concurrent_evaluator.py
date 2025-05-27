#!/usr/bin/env python3
"""Concurrent evaluation system for faster medical reasoning evaluation."""
import openai
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Optional, Tuple
from tqdm.asyncio import tqdm
import pandas as pd
from datetime import datetime
import ssl

from config import OPENAI_API_KEY, OPENAI_MODELS
from data_loader import MedQADataLoader
from reasoning_prompts import PromptTemplates

# For RAG integration
# from rag_langchain import LangChainRAGEvaluator
# from rag_graphrag import GraphRAGEvaluator

class ConcurrentOpenAIEvaluator:
    """Concurrent evaluator for OpenAI models with rate limiting."""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo", 
                 max_concurrent: int = 10, 
                 requests_per_minute: int = 100,
                 langchain_rag_evaluator: Optional[object] = None,
                 graphrag_evaluator: Optional[object] = None):
        """Initialize the concurrent evaluator.
        
        Args:
            model_name: OpenAI model to use
            max_concurrent: Maximum concurrent requests
            requests_per_minute: Rate limit (requests per minute)
            langchain_rag_evaluator: RAG evaluator for langchain mode
            graphrag_evaluator: RAG evaluator for graphrag mode
        """
        if model_name not in OPENAI_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(OPENAI_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_config = OPENAI_MODELS[model_name]
        self.max_concurrent = max_concurrent
        self.requests_per_minute = requests_per_minute
        
        # Rate limiting
        self.request_interval = 60.0 / requests_per_minute  # Seconds between requests
        self.last_request_time = 0
        
        # Initialize data loader
        self.data_loader = MedQADataLoader()
        
        # Store RAG evaluators
        self.langchain_rag_evaluator = langchain_rag_evaluator
        self.graphrag_evaluator = graphrag_evaluator
        
        # Track API usage
        self.api_calls = 0
        self.total_tokens = 0
        self.failed_requests = 0
    
    async def _make_concurrent_api_call(self, session: aiohttp.ClientSession, prompt: str, semaphore: asyncio.Semaphore) -> Tuple[str, Optional[Dict]]:
        """Make a single API call with rate limiting and error handling."""
        async with semaphore:  # Limit concurrent requests
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.request_interval:
                await asyncio.sleep(self.request_interval - time_since_last)
            
            self.last_request_time = time.time()
            
            # Prepare request
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.model_config["max_tokens"],
                "temperature": self.model_config["temperature"],
                "top_p": self.model_config["top_p"]
            }
            
            # Make request with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    async with session.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers=headers,
                        json=data,
                        ssl=ssl.create_default_context()
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            self.api_calls += 1
                            
                            # Track token usage
                            if 'usage' in result:
                                self.total_tokens += result['usage']['total_tokens']
                            
                            content = result['choices'][0]['message']['content'].strip()
                            return content, None
                        
                        elif response.status == 429:  # Rate limit
                            wait_time = 2 ** attempt
                            print(f"Rate limit hit, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            
                        else:
                            error_text = await response.text()
                            if attempt == max_retries - 1:
                                self.failed_requests += 1
                                return "", {"error": f"HTTP {response.status}: {error_text}"}
                            await asyncio.sleep(2 ** attempt)
                
                except Exception as e:
                    if attempt == max_retries - 1:
                        self.failed_requests += 1
                        return "", {"error": str(e)}
                    await asyncio.sleep(2 ** attempt)
            
            return "", {"error": "Max retries exceeded"}
    
    def _extract_answer(self, response: str) -> str:
        """Extract the answer choice from the <answer> tag in model response."""
        import re
        
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
            r'(?:Therefore|Thus|Hence),?\s+(?:the\s+)?correct\s+answer\s+is\s+([A-Z])\.',
            r'(?:Answer|answer):\s*([A-Z])\b',
            r'(?:Thus|Therefore|Hence),?\s*(?:the\s+)?(?:answer\s+is\s+)?\*?\*?([A-Z])\b',
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
    
    async def _evaluate_sample_async(self, 
                                   session: aiohttp.ClientSession, 
                                   semaphore: asyncio.Semaphore, 
                                   sample: Dict, 
                                   prompt_type: str,
                                   rag_mode: Optional[str] = None,
                                   rag_params: Optional[Dict] = None) -> Dict:
        """Evaluate a single sample asynchronously, with optional RAG."""
        question = sample['Question']
        options = sample['Options']
        correct_answer = self.data_loader.get_correct_answer(sample)
        correct_choice = self.data_loader.get_answer_choice(sample)

        context: Optional[str] = None
        rag_source_info: Optional[Dict] = None
        actual_prompt_type = prompt_type
        current_rag_mode = rag_mode # To store the mode if RAG succeeded

        if rag_mode:
            rag_params = rag_params or {}
            # print(f"  Concurrent: Enhancing with {rag_mode} RAG (params: {rag_params})...") # Too verbose for concurrent
            retrieved_context_data = None
            try:
                if rag_mode == "langchain":
                    if not self.langchain_rag_evaluator:
                        from rag_langchain import LangChainRAGEvaluator
                        self.langchain_rag_evaluator = LangChainRAGEvaluator()
                    # LangChain's retrieve_context is synchronous. 
                    # Run it in a thread pool executor to avoid blocking the event loop.
                    loop = asyncio.get_running_loop()
                    retrieved_context_data = await loop.run_in_executor(
                        None, # Uses default ThreadPoolExecutor
                        self.langchain_rag_evaluator.retrieve_context,
                        question,
                        rag_params.get('k_retrieval', self.langchain_rag_evaluator.k_retrieval if hasattr(self.langchain_rag_evaluator, 'k_retrieval') else 5)
                    )
                    context = retrieved_context_data
                    rag_source_info = {"type": "langchain", "retrieved_text": context} 

                elif rag_mode == "graphrag":
                    if not self.graphrag_evaluator:
                        from rag_graphrag import GraphRAGEvaluator
                        self.graphrag_evaluator = GraphRAGEvaluator()
                    
                    retrieved_context_data = await self.graphrag_evaluator.retrieve_context(
                        question,
                        search_type=rag_params.get('graphrag_search_type', "global"),
                        k=rag_params.get('k_retrieval', 5),
                        community_level=rag_params.get('graphrag_community_level', 2)
                    )
                    context = retrieved_context_data
                    rag_source_info = {"type": "graphrag", "search_type": rag_params.get('graphrag_search_type', "global"), "retrieved_text": context}
                
                else:
                    # print(f"    Warning: Unknown RAG mode '{rag_mode}'. Proceeding without RAG.")
                    current_rag_mode = None 

                if context and "_rag" not in prompt_type:
                    actual_prompt_type = f"{prompt_type}_rag"
                elif not context:
                    # print(f"    Warning: RAG mode '{rag_mode}' was specified but no context was retrieved. Falling back to non-RAG.")
                    actual_prompt_type = prompt_type 
                    current_rag_mode = None 

            except Exception as e:
                # print(f"    Error during RAG context retrieval ({rag_mode}): {e}. Proceeding without RAG.")
                context = None
                rag_source_info = {"type": rag_mode, "error": str(e)}
                actual_prompt_type = prompt_type
                current_rag_mode = None
        
        # Generate prompt
        try:
            prompt = PromptTemplates.get_prompt(actual_prompt_type, question, options, context=context)
        except ValueError as e:
            # print(f"Error getting prompt for type '{actual_prompt_type}': {e}. Falling back to '{prompt_type}'.")
            actual_prompt_type = prompt_type
            prompt = PromptTemplates.get_prompt(actual_prompt_type, question, options)
        
        # Get model response
        start_time = time.time()
        response, error = await self._make_concurrent_api_call(session, prompt, semaphore)
        response_time = time.time() - start_time
        
        # Extract predicted answer
        predicted_choice = self._extract_answer(response) if not error else "ERROR"
        is_correct = predicted_choice == correct_choice
        
        return {
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'correct_choice': correct_choice,
            'prompt_type': actual_prompt_type,
            'model_response': response,
            'predicted_choice': predicted_choice,
            'is_correct': is_correct,
            'response_time': response_time,
            'specialty': self.data_loader.get_sample_specialty(sample),
            'error': error,
            'concurrent_processed': True,
            'rag_mode': current_rag_mode, 
            'rag_source_info': rag_source_info
        }
    
    async def _evaluate_batch_async(self, 
                                  data: List[Dict], 
                                  prompt_types: List[str],
                                  rag_mode: Optional[str] = None,
                                  rag_params: Optional[Dict] = None) -> List[Dict]:
        """Evaluate multiple samples concurrently, with optional RAG."""
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create all tasks
        tasks = []
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2, limit_per_host=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for prompt_type in prompt_types:
                for sample in data:
                    task = self._evaluate_sample_async(session, semaphore, sample, prompt_type, 
                                                       rag_mode=rag_mode, rag_params=rag_params)
                    tasks.append(task)
            
            # Run all tasks with progress bar
            print(f"🚀 Running {len(tasks)} requests concurrently (max {self.max_concurrent} at once)")
            print(f"⚡ Rate limit: {self.requests_per_minute} requests/minute")
            
            # Use asyncio.gather with progress tracking
            results = []
            completed = 0
            total = len(tasks)
            
            # Create progress bar
            from tqdm import tqdm
            pbar = tqdm(total=total, desc="Processing requests")
            
            # Process tasks in batches to avoid overwhelming the system
            batch_size = min(self.max_concurrent * 2, len(tasks))
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        # Handle exceptions by creating error result
                        error_result = {
                            'question': 'ERROR',
                            'options': {},
                            'correct_answer': 'ERROR',
                            'correct_choice': 'ERROR',
                            'prompt_type': 'unknown',
                            'model_response': str(result),
                            'predicted_choice': 'ERROR',
                            'is_correct': False,
                            'response_time': 0,
                            'specialty': 'unknown',
                            'error': {'error': str(result)},
                            'concurrent_processed': True
                        }
                        results.append(error_result)
                    else:
                        results.append(result)
                    
                    completed += 1
                    pbar.update(1)
            
            pbar.close()
        
        return results
    
    def evaluate_dataset_concurrent(self, 
                                  split: str = "test", 
                                  prompt_types: List[str] = ["direct"],
                                  sample_size: Optional[int] = None,
                                  specialty_filter: Optional[str] = None,
                                  save_results: bool = True,
                                  rag_mode: Optional[str] = None,
                                  rag_params: Optional[Dict] = None) -> Dict:
        """Evaluate the model on a dataset split using concurrent requests, with optional RAG."""
        
        print(f"Starting CONCURRENT evaluation on {split} split")
        print(f"Model: {self.model_name}")
        print(f"Prompt types: {prompt_types}")
        print(f"Max concurrent: {self.max_concurrent}, RPM: {self.requests_per_minute}")
        if rag_mode:
            print(f"RAG Mode: {rag_mode}")
            if rag_params:
                print(f"RAG Params: {rag_params}")

        # Initialize RAG evaluators if mode is set and they are not pre-initialized
        if rag_mode == "langchain" and not self.langchain_rag_evaluator:
            from rag_langchain import LangChainRAGEvaluator
            self.langchain_rag_evaluator = LangChainRAGEvaluator()
            print(f"Initialized LangChainRAGEvaluator for concurrent dataset evaluation.")
        
        if rag_mode == "graphrag" and not self.graphrag_evaluator:
            from rag_graphrag import GraphRAGEvaluator
            self.graphrag_evaluator = GraphRAGEvaluator()
            print(f"Initialized GraphRAGEvaluator for concurrent dataset evaluation.")
            # Async GraphRAG setup like prepare_documents, run_indexing might be needed.
            # For now, assuming retrieve_context handles its own setup if necessary.
            # Or, these are run externally via CLI before evaluation.

        # Get data
        data = self.data_loader.get_split(split, sample_size, specialty_filter)
        print(f"Evaluating on {len(data)} samples")
        
        # Run evaluation concurrently
        start_time = time.time()
        # Pass rag_mode and rag_params to _evaluate_batch_async
        all_results = asyncio.run(self._evaluate_batch_async(data, prompt_types, rag_mode=rag_mode, rag_params=rag_params))
        total_time = time.time() - start_time
        
        print(f"\n✅ Concurrent evaluation completed!")
        print(f"🕒 Total time: {total_time:.1f} seconds")
        print(f"📊 Successful requests: {self.api_calls}")
        print(f"❌ Failed requests: {self.failed_requests}")
        print(f"🔥 Average requests/second: {self.api_calls/total_time:.2f}")
        
        # Generate comprehensive results
        results_summary = self._generate_results_summary(all_results, split)
        
        if save_results:
            self._save_results(all_results, results_summary, split)
        
        return {
            'detailed_results': all_results,
            'summary': results_summary,
            'api_usage': {
                'total_calls': self.api_calls,
                'total_tokens': self.total_tokens,
                'failed_requests': self.failed_requests,
                'concurrent_processing': True,
                'processing_time': total_time,
                'requests_per_second': self.api_calls / total_time if total_time > 0 else 0
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
                'timestamp': datetime.now().isoformat(),
                'concurrent_processing': True,
                'max_concurrent': self.max_concurrent,
                'requests_per_minute': self.requests_per_minute
            },
            'overall_performance': {},
            'performance_by_prompt': {},
            'performance_by_specialty': {},
            'performance_by_rag_mode': {},
            'error_analysis': {}
        }
        
        # Filter out failed requests for accuracy calculation
        successful_df = df[df['predicted_choice'] != 'ERROR']
        
        if len(successful_df) > 0:
            # Overall performance
            overall_accuracy = float(successful_df['is_correct'].mean())
            summary['overall_performance']['accuracy'] = overall_accuracy
            summary['overall_performance']['correct_count'] = int(successful_df['is_correct'].sum())
            summary['overall_performance']['total_count'] = len(successful_df)
            summary['overall_performance']['avg_response_time'] = float(successful_df['response_time'].mean())
            
            # Performance by prompt type
            for prompt_type in successful_df['prompt_type'].unique():
                prompt_df = successful_df[successful_df['prompt_type'] == prompt_type]
                summary['performance_by_prompt'][prompt_type] = {
                    'accuracy': float(prompt_df['is_correct'].mean()),
                    'correct_count': int(prompt_df['is_correct'].sum()),
                    'total_count': len(prompt_df),
                    'avg_response_time': float(prompt_df['response_time'].mean())
                }
            
            # Performance by specialty
            for specialty in successful_df['specialty'].unique():
                specialty_df = successful_df[successful_df['specialty'] == specialty]
                summary['performance_by_specialty'][specialty] = {
                    'accuracy': float(specialty_df['is_correct'].mean()),
                    'correct_count': int(specialty_df['is_correct'].sum()),
                    'total_count': len(specialty_df)
                }
            
            # Performance by RAG mode
            if 'rag_mode' in successful_df.columns:
                summary['performance_by_rag_mode'] = {}
                for mode_val in successful_df['rag_mode'].fillna('None').unique(): # mode is a built-in, use mode_val
                    mode_df = successful_df[successful_df['rag_mode'].fillna('None') == mode_val]
                    summary['performance_by_rag_mode'][mode_val] = {
                        'accuracy': float(mode_df['is_correct'].mean() if not mode_df.empty else 0),
                        'correct_count': int(mode_df['is_correct'].sum()),
                        'total_count': len(mode_df),
                        'avg_response_time': float(mode_df['response_time'].mean() if not mode_df.empty else 0)
                    }
        
        # Error analysis
        summary['error_analysis']['total_errors'] = int((df['predicted_choice'] == 'ERROR').sum())
        summary['error_analysis']['unknown_answers'] = int((df['predicted_choice'] == 'UNKNOWN').sum())
        summary['error_analysis']['failed_requests'] = self.failed_requests
        
        # Error analysis by RAG mode
        # Ensure incorrect_df is defined earlier if not already
        incorrect_df = df[~df['is_correct']]
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
        detailed_file = f"results/{self.model_name}_{split}_{timestamp}_concurrent_detailed.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_file = f"results/{self.model_name}_{split}_{timestamp}_concurrent_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = f"results/{self.model_name}_{split}_{timestamp}_concurrent.csv"
        pd.DataFrame(results).to_csv(csv_file, index=False)
        
        print(f"\n📁 Results saved:")
        print(f"  Detailed: {detailed_file}")
        print(f"  Summary: {summary_file}")
        print(f"  CSV: {csv_file}")

# Example usage
if __name__ == "__main__":
    # Initialize concurrent evaluator
    evaluator = ConcurrentOpenAIEvaluator("gpt-4o-mini", max_concurrent=5, requests_per_minute=60)
    
    # Run concurrent evaluation
    print("Running concurrent evaluation test...")
    results = evaluator.evaluate_dataset_concurrent(
        split="test_filtered_6",
        prompt_types=["direct", "chain_of_thought"],
        sample_size=10,
        save_results=True
    )
    
    print(f"\nConcurrent evaluation completed!")
    print(f"Overall accuracy: {results['summary']['overall_performance']['accuracy']:.3f}")
    print(f"API calls made: {results['api_usage']['total_calls']}")
    print(f"Processing time: {results['api_usage']['processing_time']:.1f} seconds")
    print(f"Requests/second: {results['api_usage']['requests_per_second']:.2f}") 