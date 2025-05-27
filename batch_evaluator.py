"""Batch evaluation system for medical reasoning tasks using OpenAI Batch API."""
import openai
import json
import time
import os
import uuid
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from config import OPENAI_API_KEY, OPENAI_MODELS, BATCH_SETTINGS
from data_loader import MedQADataLoader
from reasoning_prompts import PromptTemplates

class OpenAIBatchEvaluator:
    """Batch evaluator for OpenAI models on medical reasoning tasks."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the batch evaluator with a specific model."""
        if model_name not in OPENAI_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(OPENAI_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_config = OPENAI_MODELS[model_name]
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize data loader
        self.data_loader = MedQADataLoader()
        
        # Batch settings
        self.batch_dir = BATCH_SETTINGS["batch_dir"]
        self.poll_interval = BATCH_SETTINGS["poll_interval"]
        if BATCH_SETTINGS.get("fast_poll", False):
            self.poll_interval = 10  # Faster polling for development
        self.max_wait_time = BATCH_SETTINGS["max_wait_time"]
        self.auto_cleanup = BATCH_SETTINGS["auto_cleanup"]
        self.demo_mode = BATCH_SETTINGS.get("demo_mode", False)
        
        # Create batch directory
        os.makedirs(self.batch_dir, exist_ok=True)
        
        # Track batch processing
        self.batch_jobs = []
        self.total_api_calls = 0
        self.total_tokens = 0
    
    def _create_batch_request(self, sample: Dict, prompt_type: str, custom_id: str) -> Dict:
        """Create a single batch request for a sample."""
        question = sample['Question']
        options = sample['Options']
        
        # Generate prompt
        prompt = PromptTemplates.get_prompt(prompt_type, question, options)
        
        # Create batch request format
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.model_config["max_tokens"],
                "temperature": self.model_config["temperature"],
                "top_p": self.model_config["top_p"]
            }
        }
    
    def _create_batch_file(self, data: List[Dict], prompt_types: List[str]) -> Tuple[str, Dict]:
        """Create a JSONL batch file for the samples."""
        batch_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = f"{self.batch_dir}/batch_{self.model_name}_{timestamp}_{batch_id}.jsonl"
        
        requests = []
        request_metadata = {}
        
        for prompt_type in prompt_types:
            for idx, sample in enumerate(data):
                custom_id = f"{prompt_type}_{idx}_{batch_id}"
                
                # Store metadata for result processing
                request_metadata[custom_id] = {
                    'sample_index': idx,
                    'sample': sample,
                    'prompt_type': prompt_type,
                    'correct_answer': self.data_loader.get_correct_answer(sample),
                    'correct_choice': self.data_loader.get_answer_choice(sample),
                    'specialty': self.data_loader.get_sample_specialty(sample)
                }
                
                # Create batch request
                request = self._create_batch_request(sample, prompt_type, custom_id)
                requests.append(request)
        
        # Write JSONL file
        with open(batch_filename, 'w', encoding='utf-8') as f:
            for request in requests:
                f.write(json.dumps(request) + '\n')
        
        print(f"📁 Created batch file: {batch_filename}")
        print(f"📊 Total requests: {len(requests)}")
        
        return batch_filename, request_metadata
    
    def _upload_batch_file(self, batch_filename: str) -> str:
        """Upload batch file to OpenAI."""
        print(f"☁️  Uploading batch file...")
        
        with open(batch_filename, 'rb') as f:
            batch_file = self.client.files.create(
                file=f,
                purpose="batch"
            )
        
        print(f"✅ Uploaded file ID: {batch_file.id}")
        return batch_file.id
    
    def _submit_batch_job(self, file_id: str) -> str:
        """Submit batch job to OpenAI."""
        print(f"🚀 Submitting batch job...")
        
        batch_job = self.client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        
        print(f"✅ Batch job submitted: {batch_job.id}")
        print(f"🕒 Status: {batch_job.status}")
        
        return batch_job.id
    
    def _poll_batch_status(self, batch_id: str) -> Dict:
        """Poll batch job status until completion."""
        print(f"\n🔄 Monitoring batch job: {batch_id}")
        print(f"⏱️  Polling every {self.poll_interval} seconds...")
        
        start_time = time.time()
        
        with tqdm(desc="Waiting for completion", unit="checks") as pbar:
            while True:
                # Check if we've exceeded max wait time
                elapsed = time.time() - start_time
                if elapsed > self.max_wait_time:
                    raise TimeoutError(f"Batch job exceeded maximum wait time of {self.max_wait_time} seconds")
                
                # Get batch status
                batch_job = self.client.batches.retrieve(batch_id)
                status = batch_job.status
                
                pbar.set_description(f"Status: {status}")
                pbar.update(1)
                
                if status == "completed":
                    print(f"\n✅ Batch job completed!")
                    print(f"🕒 Total wait time: {elapsed:.1f} seconds")
                    return batch_job
                
                elif status == "failed":
                    print(f"\n❌ Batch job failed!")
                    if hasattr(batch_job, 'errors') and batch_job.errors:
                        print(f"Errors: {batch_job.errors}")
                    raise RuntimeError(f"Batch job {batch_id} failed")
                
                elif status == "cancelled":
                    raise RuntimeError(f"Batch job {batch_id} was cancelled")
                
                elif status in ["validating", "in_progress", "finalizing"]:
                    # Job is still processing
                    time.sleep(self.poll_interval)
                    continue
                
                else:
                    print(f"⚠️  Unknown status: {status}")
                    time.sleep(self.poll_interval)
                    continue
    
    def _download_results(self, batch_job) -> List[Dict]:
        """Download and parse batch results."""
        print(f"📥 Downloading results...")
        
        # Download output file
        if not batch_job.output_file_id:
            raise RuntimeError("No output file available")
        
        result_content = self.client.files.content(batch_job.output_file_id).content
        
        # Parse results
        results = []
        for line in result_content.decode('utf-8').strip().split('\n'):
            if line:
                results.append(json.loads(line))
        
        print(f"✅ Downloaded {len(results)} results")
        
        # Update API usage tracking
        self.total_api_calls += len(results)
        if hasattr(batch_job, 'request_counts'):
            self.total_api_calls = batch_job.request_counts.completed
        
        return results
    
    def _extract_answer(self, response: str) -> str:
        """Extract the answer choice (A, B, C, D, etc.) from model response."""
        import re
        
        # Look for patterns like "Answer: A", "A)", "(A)", or standalone "A"
        patterns = [
            r'(?:Answer|answer):\s*([A-Z])',
            r'(?:Answer|answer)\s*([A-Z])',
            r'([A-Z])\)',
            r'\(([A-Z])\)',
            r'^([A-Z])$',
            r'\b([A-Z])\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.strip())
            if match:
                return match.group(1).upper()
        
        # If no clear pattern found, look for the last capital letter
        letters = re.findall(r'[A-Z]', response)
        if letters:
            return letters[-1]
        
        return "UNKNOWN"
    
    def _process_batch_results(self, results: List[Dict], request_metadata: Dict) -> List[Dict]:
        """Process batch results and create evaluation records."""
        processed_results = []
        
        print(f"🔍 Processing {len(results)} batch results...")
        
        for result in tqdm(results, desc="Processing results"):
            custom_id = result['custom_id']
            
            if custom_id not in request_metadata:
                print(f"⚠️  Warning: Unknown custom_id {custom_id}")
                continue
            
            metadata = request_metadata[custom_id]
            
            # Extract response
            if 'response' in result and 'body' in result['response']:
                response_content = result['response']['body']['choices'][0]['message']['content']
                predicted_choice = self._extract_answer(response_content)
                is_correct = predicted_choice == metadata['correct_choice']
                
                # Calculate estimated response time (batch doesn't provide real response times)
                estimated_response_time = 1.0  # Default estimate
                
                processed_result = {
                    'question': metadata['sample']['Question'],
                    'options': metadata['sample']['Options'],
                    'correct_answer': metadata['correct_answer'],
                    'correct_choice': metadata['correct_choice'],
                    'prompt_type': metadata['prompt_type'],
                    'model_response': response_content,
                    'predicted_choice': predicted_choice,
                    'is_correct': is_correct,
                    'response_time': estimated_response_time,
                    'specialty': metadata['specialty'],
                    'batch_processed': True
                }
                
                processed_results.append(processed_result)
                
            else:
                print(f"⚠️  Error in result for {custom_id}: {result.get('error', 'Unknown error')}")
        
        print(f"✅ Successfully processed {len(processed_results)} results")
        return processed_results
    
    def _cleanup_files(self, batch_filename: str, file_id: str):
        """Clean up batch files if auto_cleanup is enabled."""
        if not self.auto_cleanup:
            return
        
        print(f"🧹 Cleaning up files...")
        
        # Delete local file
        try:
            os.remove(batch_filename)
            print(f"🗑️  Deleted local file: {batch_filename}")
        except Exception as e:
            print(f"⚠️  Could not delete local file: {e}")
        
        # Delete uploaded file
        try:
            self.client.files.delete(file_id)
            print(f"🗑️  Deleted uploaded file: {file_id}")
        except Exception as e:
            print(f"⚠️  Could not delete uploaded file: {e}")
    
    def evaluate_dataset_batch(self, 
                              split: str = "test", 
                              prompt_types: List[str] = ["direct"],
                              sample_size: Optional[int] = None,
                              specialty_filter: Optional[str] = None,
                              save_results: bool = True,
                              use_batch: bool = True) -> Dict:
        """Evaluate the model on a dataset split using batch processing."""
        
        print(f"🚀 Starting Batch Evaluation")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Split: {split}")
        print(f"Prompt types: {prompt_types}")
        
        # Get data
        data = self.data_loader.get_split(split, sample_size, specialty_filter)
        total_requests = len(data) * len(prompt_types)
        
        print(f"📊 Evaluating on {len(data)} samples")
        print(f"📊 Total requests: {total_requests}")
        
        # Check if batch processing should be used
        min_batch_size = BATCH_SETTINGS["min_batch_size"]
        if not use_batch or total_requests < min_batch_size:
            print(f"⚠️  Using synchronous processing (requests: {total_requests} < min batch: {min_batch_size})")
            # Fall back to regular evaluator
            from model_evaluator import OpenAIEvaluator
            sync_evaluator = OpenAIEvaluator(self.model_name)
            return sync_evaluator.evaluate_dataset(split, prompt_types, sample_size, specialty_filter, save_results)
        
        # Demo mode for testing
        if self.demo_mode:
            print(f"\n🎭 DEMO MODE: Simulating batch processing...")
            print(f"💡 In real mode, this would submit {total_requests} requests to OpenAI Batch API")
            print(f"💰 Estimated cost savings: ~50% vs synchronous")
            print(f"⏱️  Estimated completion time: 10 minutes - 24 hours")
            
            # Simulate batch processing by using sync processing but marking as batch
            from model_evaluator import OpenAIEvaluator
            sync_evaluator = OpenAIEvaluator(self.model_name)
            results = sync_evaluator.evaluate_dataset(split, prompt_types, sample_size, specialty_filter, save_results)
            
            # Mark as batch processed and adjust response
            results['api_usage']['batch_processing'] = True
            results['api_usage']['processing_time'] = results['api_usage'].get('total_calls', 0) * 0.1  # Simulate batch time
            results['summary']['evaluation_info']['batch_processing'] = True
            
            # Add batch processing markers to detailed results
            for result in results['detailed_results']:
                result['batch_processed'] = True
            
            print(f"\n🎭 Demo completed! In real batch mode:")
            print(f"   • Requests would be processed concurrently")
            print(f"   • Cost would be ~50% lower")
            print(f"   • Processing time: varies (10min - 24hrs)")
            
            return results
        
        try:
            start_time = time.time()
            
            # Step 1: Create batch file
            print(f"\n📝 Step 1: Creating batch file...")
            batch_filename, request_metadata = self._create_batch_file(data, prompt_types)
            
            # Step 2: Upload file
            print(f"\n☁️  Step 2: Uploading batch file...")
            file_id = self._upload_batch_file(batch_filename)
            
            # Step 3: Submit batch job
            print(f"\n🚀 Step 3: Submitting batch job...")
            batch_id = self._submit_batch_job(file_id)
            
            # Step 4: Wait for completion
            print(f"\n⏳ Step 4: Waiting for completion...")
            print(f"💡 TIP: Batch jobs can take 10 minutes to 24 hours")
            print(f"💡 TIP: You can check status later with: python -c \"from batch_evaluator import *; check_batch_status('{batch_id}')\"")
            batch_job = self._poll_batch_status(batch_id)
            
            # Step 5: Download results
            print(f"\n📥 Step 5: Downloading results...")
            raw_results = self._download_results(batch_job)
            
            # Step 6: Process results
            print(f"\n🔍 Step 6: Processing results...")
            processed_results = self._process_batch_results(raw_results, request_metadata)
            
            # Step 7: Generate summary
            print(f"\n📊 Step 7: Generating summary...")
            results_summary = self._generate_results_summary(processed_results, split)
            
            # Step 8: Save results
            if save_results:
                print(f"\n💾 Step 8: Saving results...")
                self._save_results(processed_results, results_summary, split)
            
            # Step 9: Cleanup
            print(f"\n🧹 Step 9: Cleanup...")
            self._cleanup_files(batch_filename, file_id)
            
            total_time = time.time() - start_time
            print(f"\n🎉 Batch evaluation completed!")
            print(f"🕒 Total time: {total_time:.1f} seconds")
            print(f"💰 Estimated cost savings: ~50% vs synchronous")
            
            return {
                'detailed_results': processed_results,
                'summary': results_summary,
                'api_usage': {
                    'total_calls': self.total_api_calls,
                    'total_tokens': self.total_tokens,
                    'batch_processing': True,
                    'processing_time': total_time
                }
            }
            
        except Exception as e:
            print(f"\n❌ Batch evaluation failed: {e}")
            print(f"💡 Falling back to synchronous processing...")
            
            # Cleanup on error
            try:
                if 'batch_filename' in locals():
                    self._cleanup_files(batch_filename, file_id if 'file_id' in locals() else None)
            except:
                pass
            
            # Fall back to regular evaluator
            from model_evaluator import OpenAIEvaluator
            sync_evaluator = OpenAIEvaluator(self.model_name)
            return sync_evaluator.evaluate_dataset(split, prompt_types, sample_size, specialty_filter, save_results)
    
    def _generate_results_summary(self, results: List[Dict], split: str) -> Dict:
        """Generate comprehensive results summary."""
        df = pd.DataFrame(results)
        
        summary = {
            'evaluation_info': {
                'model': self.model_name,
                'split': split,
                'total_samples': len(results),
                'timestamp': datetime.now().isoformat(),
                'batch_processing': True
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
            errors_by_specialty = incorrect_df['specialty'].value_counts().to_dict()
            errors_by_prompt = incorrect_df['prompt_type'].value_counts().to_dict()
            
            summary['error_analysis']['errors_by_specialty'] = {k: int(v) for k, v in errors_by_specialty.items()}
            summary['error_analysis']['errors_by_prompt'] = {k: int(v) for k, v in errors_by_prompt.items()}
        
        return summary
    
    def _save_results(self, results: List[Dict], summary: Dict, split: str) -> None:
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Save detailed results
        detailed_file = f"results/{self.model_name}_{split}_{timestamp}_batch_detailed.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_file = f"results/{self.model_name}_{split}_{timestamp}_batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = f"results/{self.model_name}_{split}_{timestamp}_batch.csv"
        pd.DataFrame(results).to_csv(csv_file, index=False)
        
        print(f"📁 Results saved:")
        print(f"  Detailed: {detailed_file}")
        print(f"  Summary: {summary_file}")
        print(f"  CSV: {csv_file}")

def check_batch_status(batch_id: str) -> None:
    """Utility function to check the status of a batch job."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        batch_job = client.batches.retrieve(batch_id)
        print(f"📊 Batch Job Status: {batch_id}")
        print(f"Status: {batch_job.status}")
        
        if hasattr(batch_job, 'request_counts'):
            counts = batch_job.request_counts
            print(f"Requests - Total: {counts.total}, Completed: {counts.completed}, Failed: {counts.failed}")
        
        if batch_job.status == "completed":
            print(f"✅ Batch completed! Output file: {batch_job.output_file_id}")
        elif batch_job.status == "failed":
            print(f"❌ Batch failed!")
            if hasattr(batch_job, 'errors'):
                print(f"Errors: {batch_job.errors}")
        else:
            print(f"⏳ Still processing...")
            
    except Exception as e:
        print(f"❌ Error checking batch status: {e}")

def list_batch_jobs() -> None:
    """List recent batch jobs."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        batches = client.batches.list(limit=10)
        print(f"📋 Recent Batch Jobs:")
        
        for batch in batches.data:
            print(f"  {batch.id}: {batch.status} ({batch.endpoint})")
            
    except Exception as e:
        print(f"❌ Error listing batches: {e}")

def cancel_batch_job(batch_id: str) -> None:
    """Cancel a batch job."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        batch_job = client.batches.cancel(batch_id)
        print(f"🛑 Cancelled batch job: {batch_id}")
        print(f"Status: {batch_job.status}")
        
    except Exception as e:
        print(f"❌ Error cancelling batch: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize batch evaluator
    evaluator = OpenAIBatchEvaluator("gpt-4o-mini")
    
    # Run batch evaluation
    print("Running batch evaluation test...")
    results = evaluator.evaluate_dataset_batch(
        split="test_filtered_6",
        prompt_types=["direct", "chain_of_thought"],
        sample_size=20,  # Small sample for testing
        save_results=True
    )
    
    print(f"\nBatch evaluation completed!")
    print(f"Overall accuracy: {results['summary']['overall_performance']['accuracy']:.3f}")
    print(f"API calls made: {results['api_usage']['total_calls']}")
    print(f"Processing time: {results['api_usage']['processing_time']:.1f} seconds") 