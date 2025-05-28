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

# For RAG integration
# from rag_langchain import LangChainRAGEvaluator
# from rag_graphrag import GraphRAGEvaluator

class OpenAIBatchEvaluator:
    """Batch evaluator for OpenAI models on medical reasoning tasks."""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 langchain_rag_evaluator: Optional[object] = None,
                 graphrag_evaluator: Optional[object] = None):
        """Initialize the batch evaluator with a specific model and optional RAG evaluators."""
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
    
    def _create_batch_request(self, 
                              sample: Dict, 
                              prompt_type: str, 
                              custom_id: str,
                              context: Optional[str] = None) -> Dict:
        """Create a single batch request for a sample, with optional RAG context."""
        question = sample['Question']
        options = sample['Options']
        
        # Generate prompt - use context if provided
        prompt = PromptTemplates.get_prompt(prompt_type, question, options, context=context)
        
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
                "top_p": self.model_config["top_p"],
                "frequency_penalty": self.model_config.get("frequency_penalty", 0.0),
                "presence_penalty": self.model_config.get("presence_penalty", 0.0),
                "seed": self.model_config.get("seed", 42)
            }
        }
    
    def _create_batch_file(self, 
                           data: List[Dict], 
                           prompt_types: List[str],
                           rag_mode: Optional[str] = None,
                           rag_params: Optional[Dict] = None) -> Tuple[str, Dict]:
        """Create a JSONL batch file, retrieving RAG context if enabled."""
        batch_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = f"{self.batch_dir}/batch_{self.model_name}_{timestamp}_{batch_id}.jsonl"
        
        requests = []
        request_metadata = {}
        
        # Initialize RAG evaluators on demand if not passed in constructor
        if rag_mode == "langchain" and not self.langchain_rag_evaluator:
            from rag_langchain import LangChainRAGEvaluator
            self.langchain_rag_evaluator = LangChainRAGEvaluator()
            print(f"    Initialized LangChainRAGEvaluator for batch prep.")
        
        if rag_mode == "graphrag" and not self.graphrag_evaluator:
            from rag_graphrag import GraphRAGEvaluator
            self.graphrag_evaluator = GraphRAGEvaluator()
            print(f"    Initialized GraphRAGEvaluator for batch prep.")

        for prompt_type_orig in prompt_types:
            for idx, sample in enumerate(data):
                custom_id = f"{prompt_type_orig}_{idx}_{batch_id}"
                question = sample['Question'] # Needed for RAG context retrieval

                current_context: Optional[str] = None
                current_rag_source_info: Optional[Dict] = None
                actual_prompt_type_for_request = prompt_type_orig
                current_rag_mode_for_sample = None # Tracks if RAG was successfully used for this sample

                if rag_mode:
                    effective_rag_params = rag_params or {}
                    # print(f"  Batch Prep: Enhancing sample {idx} with {rag_mode} RAG...") # Can be verbose
                    try:
                        if rag_mode == "langchain":
                            if not self.langchain_rag_evaluator: raise ValueError("Langchain RAG evaluator not initialized")
                            current_context = self.langchain_rag_evaluator.retrieve_context(
                                question,
                                k=effective_rag_params.get('k_retrieval', self.langchain_rag_evaluator.k_retrieval if hasattr(self.langchain_rag_evaluator, 'k_retrieval') else 5)
                            )
                            current_rag_source_info = {"type": "langchain", "retrieved_text": current_context}
                        
                        elif rag_mode == "graphrag":
                            if not self.graphrag_evaluator: raise ValueError("GraphRAG evaluator not initialized")
                            import asyncio
                            # GraphRAG is async, run its retrieve_context method
                            try:
                                loop = asyncio.get_running_loop()
                            except RuntimeError:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                            
                            current_context = loop.run_until_complete(
                                self.graphrag_evaluator.retrieve_context(
                                    question,
                                    search_type=effective_rag_params.get('graphrag_search_type', "global"),
                                    k=effective_rag_params.get('k_retrieval', 5),
                                    community_level=effective_rag_params.get('graphrag_community_level', 2)
                                )
                            )
                            current_rag_source_info = {"type": "graphrag", "search_type": effective_rag_params.get('graphrag_search_type', "global"), "retrieved_text": current_context}

                        if current_context and "_rag" not in actual_prompt_type_for_request:
                            actual_prompt_type_for_request = f"{prompt_type_orig}_rag"
                            current_rag_mode_for_sample = rag_mode # RAG was successful for this sample
                        elif not current_context:
                            # print(f"    Warning: RAG mode '{rag_mode}' but no context for sample {idx}. Using non-RAG prompt.")
                            actual_prompt_type_for_request = prompt_type_orig
                            current_rag_mode_for_sample = None # RAG failed for this sample

                    except Exception as e:
                        print(f"    Error during RAG context retrieval for sample {idx} ({rag_mode}): {e}. Using non-RAG prompt.")
                        current_context = None
                        current_rag_source_info = {"type": rag_mode, "error": str(e)}
                        actual_prompt_type_for_request = prompt_type_orig
                        current_rag_mode_for_sample = None # RAG failed

                # Generate the prompt (needed for both metadata and batch request)
                try:
                    prompt = PromptTemplates.get_prompt(actual_prompt_type_for_request, question, sample['Options'], context=current_context)
                except ValueError as e:
                    print(f"Error getting prompt for type '{actual_prompt_type_for_request}': {e}. Falling back to '{prompt_type_orig}'.")
                    actual_prompt_type_for_request = prompt_type_orig
                    prompt = PromptTemplates.get_prompt(actual_prompt_type_for_request, question, sample['Options'])

                # Store metadata for result processing
                request_metadata[custom_id] = {
                    'sample_index': idx,
                    'sample': sample,
                    'prompt_type': actual_prompt_type_for_request, # Log the prompt type used (RAG or not)
                    'full_prompt': prompt,  # Store the complete prompt for saving in results
                    'correct_answer': self.data_loader.get_correct_answer(sample),
                    'correct_choice': self.data_loader.get_answer_choice(sample),
                    'specialty': self.data_loader.get_sample_specialty(sample),
                    'rag_mode': current_rag_mode_for_sample, # Log if RAG was effectively used
                    'rag_source_info': current_rag_source_info # Log context or error
                }
                
                # Create batch request, passing the retrieved context (if any)
                request = self._create_batch_request(sample, actual_prompt_type_for_request, custom_id, context=current_context)
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
    
    def _process_batch_results(self, results: List[Dict], request_metadata: Dict) -> List[Dict]:
        """Process batch results and create evaluation records."""
        processed_results = []
        
        print(f"🔍 Processing {len(results)} batch results...")
        
        for result in tqdm(results, desc="Processing results"):
            custom_id = result.get('custom_id')
            response_body = result.get('response', {}).get('body', {})
            error_body = result.get('error') # OpenAI Batch API error structure

            metadata = request_metadata.get(custom_id)
            if not metadata:
                print(f"Warning: No metadata found for custom_id {custom_id}")
                # Create a minimal error entry
                error_entry = {
                    'question': "Unknown (metadata missing)",
                    'options': {},
                    'correct_answer': "N/A",
                    'correct_choice': "N/A",
                    'prompt_type': "unknown",
                    'full_prompt': "N/A (metadata missing)",  # Include full_prompt field
                    'model_response': f"Error: No metadata for custom_id {custom_id}",
                    'predicted_choice': "ERROR",
                    'is_correct': False,
                    'response_time': 0, # Batch API doesn't provide per-item response time
                    'specialty': "unknown",
                    'error': {"code": result.get('response', {}).get('status_code'), "message": "Metadata missing"},
                    'batch_processed': True,
                    'rag_mode': None,
                    'rag_source_info': None
                }
                processed_results.append(error_entry)
                continue

            response_content = ""
            predicted_choice = "ERROR"
            is_correct = False
            api_error = None

            if error_body:
                response_content = f"Batch API Error: {error_body.get('message', 'Unknown error')}"
                api_error = error_body
            elif response_body and 'choices' in response_body and response_body['choices']:
                response_content = response_body['choices'][0]['message']['content'].strip()
                predicted_choice = self._extract_answer(response_content)
                is_correct = predicted_choice == metadata['correct_choice']
                # Token usage for batch API is typically per-batch, not per-item from response
                # self.total_tokens += response_body.get('usage', {}).get('total_tokens', 0)
            else:
                response_content = "Error: Empty or malformed response from Batch API"
                api_error = {"message": response_content, "code": result.get('response', {}).get('status_code')}

            processed_results.append({
                'question': metadata['sample']['Question'],
                'options': metadata['sample']['Options'],
                'correct_answer': metadata['correct_answer'],
                'correct_choice': metadata['correct_choice'],
                'prompt_type': metadata['prompt_type'], # This now reflects if RAG prompt was used
                'full_prompt': metadata['full_prompt'],  # Include the complete prompt
                'model_response': response_content,
                'predicted_choice': predicted_choice,
                'is_correct': is_correct,
                'response_time': 0,  # Not available per item in batch
                'specialty': metadata['specialty'],
                'error': api_error,
                'batch_processed': True,
                'custom_id': custom_id,
                # Add RAG info from metadata
                'rag_mode': metadata.get('rag_mode'),
                'rag_source_info': metadata.get('rag_source_info')
            })
        
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
                              use_batch: bool = True,
                              rag_mode: Optional[str] = None,
                              rag_params: Optional[Dict] = None) -> Dict:
        """Evaluate the model on a dataset split using the Batch API, with optional RAG."""
        
        print(f"Starting BATCH evaluation on {split} split")
        print("=" * 50)
        print(f"Model: {self.model_name}")
        print(f"Prompt types: {prompt_types}")
        if rag_mode:
            print(f"RAG Mode: {rag_mode}")
            if rag_params:
                print(f"RAG Params: {rag_params}")

        # Potentially initialize RAG evaluators if not done in __init__ and rag_mode is set
        # This is mainly for the _create_batch_file step
        if rag_mode == "langchain" and not self.langchain_rag_evaluator:
            from rag_langchain import LangChainRAGEvaluator
            self.langchain_rag_evaluator = LangChainRAGEvaluator()
            print(f"Initialized LangChainRAGEvaluator for batch evaluation.")
        
        if rag_mode == "graphrag" and not self.graphrag_evaluator:
            from rag_graphrag import GraphRAGEvaluator
            self.graphrag_evaluator = GraphRAGEvaluator()
            print(f"Initialized GraphRAGEvaluator for batch evaluation.")

        # Get data
        data = self.data_loader.get_split(split, sample_size, specialty_filter)
        # Sort data by question text to ensure consistent processing order
        data = sorted(data, key=lambda x: x['Question'])
        print(f"Evaluating on {len(data)} samples")

        if self.demo_mode:
            print("\n⚠️  DEMO MODE ENABLED FOR BATCH EVALUATOR ⚠️")
            print("Simulating batch processing with synchronous calls.")
            # In demo mode, we simulate batch by calling a synchronous evaluator.
            # We need a synchronous evaluator instance here.
            # For RAG, this sync evaluator also needs RAG capabilities.
            from model_evaluator import OpenAIEvaluator # Assuming this is the sync one
            
            sync_evaluator_params = {"model_name": self.model_name}
            if rag_mode == "langchain": sync_evaluator_params["langchain_rag_evaluator"] = self.langchain_rag_evaluator
            if rag_mode == "graphrag": sync_evaluator_params["graphrag_evaluator"] = self.graphrag_evaluator
            sync_evaluator = OpenAIEvaluator(**sync_evaluator_params)

            all_results = []
            for prompt_type in prompt_types:
                for sample_item in tqdm(data, desc=f"Demo Batch ({prompt_type})"):
                    # Pass rag_mode and rag_params to the synchronous evaluator's sample evaluation
                    result = sync_evaluator.evaluate_sample(sample_item, prompt_type, rag_mode=rag_mode, rag_params=rag_params)
                    result['batch_processed'] = True # Mark as batch processed for consistency
                    result['batch_demo_simulated'] = True
                    all_results.append(result)
            
            # Update API usage from the sync evaluator if possible (it tracks its own)
            self.total_api_calls = sync_evaluator.api_calls
            self.total_tokens = sync_evaluator.total_tokens

        else:
            # Create batch file (this now handles RAG context retrieval)
            batch_filename, request_metadata = self._create_batch_file(data, prompt_types, rag_mode=rag_mode, rag_params=rag_params)
            
            # Upload batch file
            print(f"\n☁️  Step 2: Uploading batch file...")
            file_id = self._upload_batch_file(batch_filename)
            
            # Submit batch job
            print(f"\n🚀 Step 3: Submitting batch job...")
            batch_id = self._submit_batch_job(file_id)
            
            # Wait for completion
            print(f"\n⏳ Step 4: Waiting for completion...")
            print(f"💡 TIP: Batch jobs can take 10 minutes to 24 hours")
            print(f"💡 TIP: You can check status later with: python -c \"from batch_evaluator import *; check_batch_status('{batch_id}')\"")
            batch_job = self._poll_batch_status(batch_id)
            
            # Download results
            print(f"\n📥 Step 5: Downloading results...")
            raw_results = self._download_results(batch_job)
            
            # Process results
            print(f"\n🔍 Step 6: Processing results...")
            processed_results = self._process_batch_results(raw_results, request_metadata)
            
            # Generate summary
            print(f"\n📊 Step 7: Generating summary...")
            results_summary = self._generate_results_summary(processed_results, split)
            
            # Save results
            if save_results:
                print(f"\n💾 Step 8: Saving results...")
                self._save_results(processed_results, results_summary, split)
            
            # Cleanup
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
            'performance_by_rag_mode': {},
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
            for mode_val in df['rag_mode'].fillna('None').unique(): # mode is a built-in, use mode_val
                mode_df = df[df['rag_mode'].fillna('None') == mode_val]
                summary['performance_by_rag_mode'][mode_val] = {
                    'accuracy': float(mode_df['is_correct'].mean() if not mode_df.empty else 0),
                    'correct_count': int(mode_df['is_correct'].sum()),
                    'total_count': len(mode_df)
                    # Avg response time is not meaningful for batch API results per item
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