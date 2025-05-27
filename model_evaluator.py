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

class OpenAIEvaluator:
    """Evaluator for OpenAI models on medical reasoning tasks."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the evaluator with a specific model."""
        if model_name not in OPENAI_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(OPENAI_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_config = OPENAI_MODELS[model_name]
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize data loader
        self.data_loader = MedQADataLoader()
        
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
                    top_p=self.model_config["top_p"]
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
    
    def evaluate_sample(self, sample: Dict, prompt_type: str = "direct") -> Dict:
        """Evaluate a single sample."""
        question = sample['Question']
        options = sample['Options']
        correct_answer = self.data_loader.get_correct_answer(sample)
        correct_choice = self.data_loader.get_answer_choice(sample)
        
        # Generate prompt
        prompt = PromptTemplates.get_prompt(prompt_type, question, options)
        
        # Get model response
        start_time = time.time()
        response = self._make_api_call(prompt)
        response_time = time.time() - start_time
        
        # Extract predicted answer
        predicted_choice = self._extract_answer(response)
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
            'specialty': self.data_loader.get_sample_specialty(sample)
        }
    
    def evaluate_dataset(self, 
                        split: str = "test", 
                        prompt_types: List[str] = ["direct"],
                        sample_size: Optional[int] = None,
                        specialty_filter: Optional[str] = None,
                        save_results: bool = True) -> Dict:
        """Evaluate the model on a dataset split."""
        
        print(f"Starting evaluation on {split} split")
        print(f"Model: {self.model_name}")
        print(f"Prompt types: {prompt_types}")
        
        # Get data
        data = self.data_loader.get_split(split, sample_size, specialty_filter)
        print(f"Evaluating on {len(data)} samples")
        
        all_results = []
        
        for prompt_type in prompt_types:
            print(f"\nEvaluating with {prompt_type} prompts...")
            
            prompt_results = []
            for sample in tqdm(data, desc=f"Processing {prompt_type}"):
                try:
                    result = self.evaluate_sample(sample, prompt_type)
                    prompt_results.append(result)
                    all_results.append(result)
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue
            
            # Calculate accuracy for this prompt type
            correct_count = sum(1 for r in prompt_results if r['is_correct'])
            accuracy = correct_count / len(prompt_results) if prompt_results else 0
            print(f"{prompt_type} accuracy: {accuracy:.3f} ({correct_count}/{len(prompt_results)})")
        
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