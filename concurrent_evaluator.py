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

class ConcurrentOpenAIEvaluator:
    """Concurrent evaluator for OpenAI models with rate limiting."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", max_concurrent: int = 10, requests_per_minute: int = 100):
        """Initialize the concurrent evaluator.
        
        Args:
            model_name: OpenAI model to use
            max_concurrent: Maximum concurrent requests
            requests_per_minute: Rate limit (requests per minute)
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
    
    async def _evaluate_sample_async(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, 
                                   sample: Dict, prompt_type: str) -> Dict:
        """Evaluate a single sample asynchronously."""
        question = sample['Question']
        options = sample['Options']
        correct_answer = self.data_loader.get_correct_answer(sample)
        correct_choice = self.data_loader.get_answer_choice(sample)
        
        # Generate prompt
        prompt = PromptTemplates.get_prompt(prompt_type, question, options)
        
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
            'prompt_type': prompt_type,
            'model_response': response,
            'predicted_choice': predicted_choice,
            'is_correct': is_correct,
            'response_time': response_time,
            'specialty': self.data_loader.get_sample_specialty(sample),
            'error': error,
            'concurrent_processed': True
        }
    
    async def _evaluate_batch_async(self, data: List[Dict], prompt_types: List[str]) -> List[Dict]:
        """Evaluate multiple samples concurrently."""
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Create all tasks
        tasks = []
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2, limit_per_host=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for prompt_type in prompt_types:
                for sample in data:
                    task = self._evaluate_sample_async(session, semaphore, sample, prompt_type)
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
                                  save_results: bool = True) -> Dict:
        """Evaluate the model on a dataset split using concurrent processing."""
        
        print(f"🚀 Starting Concurrent Evaluation")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Split: {split}")
        print(f"Prompt types: {prompt_types}")
        print(f"Max concurrent requests: {self.max_concurrent}")
        print(f"Rate limit: {self.requests_per_minute} requests/minute")
        
        # Get data
        data = self.data_loader.get_split(split, sample_size, specialty_filter)
        total_requests = len(data) * len(prompt_types)
        
        print(f"📊 Evaluating on {len(data)} samples")
        print(f"📊 Total requests: {total_requests}")
        
        # Estimate time
        estimated_time = total_requests / self.requests_per_minute * 60
        print(f"⏰ Estimated time: {estimated_time:.1f} seconds")
        
        start_time = time.time()
        
        # Run concurrent evaluation
        results = asyncio.run(self._evaluate_batch_async(data, prompt_types))
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Concurrent evaluation completed!")
        print(f"🕒 Total time: {total_time:.1f} seconds")
        print(f"📊 Successful requests: {self.api_calls}")
        print(f"❌ Failed requests: {self.failed_requests}")
        print(f"🔥 Average requests/second: {self.api_calls/total_time:.2f}")
        
        # Generate comprehensive results
        results_summary = self._generate_results_summary(results, split)
        
        if save_results:
            self._save_results(results, results_summary, split)
        
        return {
            'detailed_results': results,
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
        
        # Error analysis
        summary['error_analysis']['total_errors'] = int((df['predicted_choice'] == 'ERROR').sum())
        summary['error_analysis']['unknown_answers'] = int((df['predicted_choice'] == 'UNKNOWN').sum())
        summary['error_analysis']['failed_requests'] = self.failed_requests
        
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