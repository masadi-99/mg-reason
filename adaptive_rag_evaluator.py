#!/usr/bin/env python3
"""
Adaptive RAG Evaluator - Two-stage RAG approach with concurrent support.

Stage 1: Ask LLM to identify k medical guidelines/knowledge areas needed to answer the question
Stage 2: Retrieve specific information for each guideline using LangChain RAG
Stage 3: Answer the question using the targeted retrievals with various reasoning modes
"""

import openai
import asyncio
import aiohttp
import json
import time
import re
import random
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from pathlib import Path

from config import OPENAI_API_KEY, OPENAI_MODELS, RANDOM_SEED
from data_loader import MedQADataLoader
from reasoning_prompts import PromptTemplates
from rag_langchain import LangChainRAGEvaluator

class AdaptiveRAGEvaluator:
    """Two-stage adaptive RAG evaluator for medical reasoning tasks with concurrent support."""
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 k_guidelines: int = 3,
                 k_retrieval_per_guideline: int = 2,
                 max_concurrent: int = 10,
                 requests_per_minute: int = 100):
        """Initialize the adaptive RAG evaluator.
        
        Args:
            model_name: OpenAI model to use
            k_guidelines: Number of medical guidelines to identify in stage 1
            k_retrieval_per_guideline: Number of documents to retrieve per guideline
            max_concurrent: Maximum concurrent requests for async processing
            requests_per_minute: Rate limit for requests per minute
        """
        if model_name not in OPENAI_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {list(OPENAI_MODELS.keys())}")
        
        # Set deterministic seeds for reproducibility
        self._set_deterministic_seeds()
        
        self.model_name = model_name
        self.model_config = OPENAI_MODELS[model_name]
        self.k_guidelines = k_guidelines
        self.k_retrieval_per_guideline = k_retrieval_per_guideline
        self.max_concurrent = max_concurrent
        self.requests_per_minute = requests_per_minute
        
        # Initialize OpenAI clients (both sync and async) with deterministic settings
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.async_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize data loader
        self.data_loader = MedQADataLoader()
        
        # Initialize LangChain RAG evaluator
        self.rag_evaluator = LangChainRAGEvaluator()
        
        # Track API usage
        self.api_calls = 0
        self.total_tokens = 0
        
        # Rate limiting
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0
        self.request_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        
        # Create guideline identification prompt
        self.guideline_prompt_template = """You are a medical expert tasked with answering a multiple-choice clinical question. To answer correctly, you must identify {k} specific medical guidelines, authoritative clinical statements, or textbook excerpts that would directly address the clinical reasoning required.

For each guideline or authoritative excerpt, generate a realistic-sounding excerpt (approximately 2-4 sentences) as if directly quoted from an official medical guideline, textbook, or clinical reference. Each excerpt should clearly and specifically address the clinical concepts or reasoning needed to answer the question correctly.

Format your response using the following tags:

<guideline_1>
[Realistic-sounding excerpt from a hypothetical authoritative medical guideline or textbook directly relevant to answering the question]
</guideline_1>

<guideline_2>
[Realistic-sounding excerpt from a hypothetical authoritative medical guideline or textbook directly relevant to answering the question]
</guideline_2>

<guideline_3>
[Realistic-sounding excerpt from a hypothetical authoritative medical guideline or textbook directly relevant to answering the question]
</guideline_3>

Question: {question}

Options:
{options}

Please generate the {k} most relevant hypothetical guideline excerpts needed to correctly answer this question.
"""
        
        # Create step-by-step guideline reasoning prompt
        self.stepwise_guideline_prompt_template = """You are a medical expert solving a clinical question step by step. For each reasoning step, you must cite a specific medical guideline excerpt that supports that step.

Break down your reasoning into exactly {k} logical steps. For each step:
1. State the reasoning step clearly
2. Provide a realistic-sounding medical guideline excerpt that supports this reasoning step

Format your response using the following structure:

<step_1>
<reasoning>Your first reasoning step</reasoning>
<guideline>Realistic excerpt from a medical guideline that supports this reasoning step</guideline>
</step_1>

<step_2>
<reasoning>Your second reasoning step</reasoning>
<guideline>Realistic excerpt from a medical guideline that supports this reasoning step</guideline>
</step_2>

<step_3>
<reasoning>Your third reasoning step</reasoning>
<guideline>Realistic excerpt from a medical guideline that supports this reasoning step</guideline>
</step_3>

Question: {question}

Options:
{options}

Please provide {k} reasoning steps with supporting medical guideline excerpts to solve this clinical question systematically.
"""
    
    def _set_deterministic_seeds(self):
        """Set all random seeds for reproducible results."""
        # Set Python random seed
        random.seed(RANDOM_SEED)
        
        # Set NumPy random seed
        np.random.seed(RANDOM_SEED)
        
        # Set Python hash seed
        os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
        
        print(f"🎯 Deterministic mode enabled with seed: {RANDOM_SEED}")
    
    def _get_deterministic_api_params(self) -> Dict:
        """Get deterministic API parameters from model config."""
        return {
            "max_tokens": self.model_config["max_tokens"],
            "temperature": self.model_config["temperature"],  # Should be 0.0 for deterministic
            "top_p": self.model_config["top_p"],  # Should be 1.0 for deterministic
            "frequency_penalty": self.model_config["frequency_penalty"],  # Should be 0.0
            "presence_penalty": self.model_config["presence_penalty"],  # Should be 0.0
            "seed": self.model_config["seed"]  # Should be 42
        }
    
    async def _make_api_call_async(self, prompt: str, max_retries: int = 3) -> str:
        """Make an async API call to OpenAI with rate limiting and retries."""
        async with self.semaphore:
            # Rate limiting
            if self.request_interval > 0:
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.request_interval:
                    await asyncio.sleep(self.request_interval - time_since_last)
                self.last_request_time = time.time()
            
            for attempt in range(max_retries):
                try:
                    # Use deterministic parameters
                    api_params = self._get_deterministic_api_params()
                    
                    response = await self.async_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        **api_params
                    )
                    
                    self.api_calls += 1
                    if hasattr(response, 'usage') and response.usage:
                        self.total_tokens += response.usage.total_tokens
                    
                    return response.choices[0].message.content.strip()
                    
                except openai.RateLimitError:
                    wait_time = (2 ** attempt) + (attempt * 0.1)  # Exponential backoff with jitter
                    print(f"Rate limit exceeded. Waiting {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                except openai.APIError as e:
                    print(f"API error on attempt {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1)
            
            raise Exception(f"Failed to get response after {max_retries} attempts")
    
    def _make_api_call(self, prompt: str, max_retries: int = 3) -> str:
        """Make a synchronous API call to OpenAI with retries."""
        for attempt in range(max_retries):
            try:
                # Use deterministic parameters
                api_params = self._get_deterministic_api_params()
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    **api_params
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
    
    def _extract_guidelines(self, response: str) -> List[str]:
        """Extract guidelines from the LLM response using regex."""
        guidelines = []
        
        # Look for guideline tags
        for i in range(1, self.k_guidelines + 1):
            pattern = f'<guideline_{i}>(.*?)</guideline_{i}>'
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                guideline_text = match.group(1).strip()
                guidelines.append(guideline_text)
            else:
                # Fallback: look for any guideline tag
                fallback_pattern = f'<guideline_{i}>(.*?)(?=<guideline_|$)'
                fallback_match = re.search(fallback_pattern, response, re.DOTALL | re.IGNORECASE)
                if fallback_match:
                    guideline_text = fallback_match.group(1).strip()
                    # Remove any closing tags
                    guideline_text = re.sub(r'</guideline_\d+>', '', guideline_text).strip()
                    guidelines.append(guideline_text)
                else:
                    guidelines.append(f"Guideline {i}: Could not extract from response")
        
        return guidelines
    
    def _extract_reasoning_steps(self, response: str) -> List[Dict]:
        """Extract reasoning steps and associated guidelines from the LLM response."""
        steps = []
        
        # Look for step tags
        for i in range(1, self.k_guidelines + 1):
            step_pattern = f'<step_{i}>(.*?)</step_{i}>'
            step_match = re.search(step_pattern, response, re.DOTALL | re.IGNORECASE)
            
            if step_match:
                step_content = step_match.group(1).strip()
                
                # Extract reasoning and guideline from step content
                reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', step_content, re.DOTALL | re.IGNORECASE)
                guideline_match = re.search(r'<guideline>(.*?)</guideline>', step_content, re.DOTALL | re.IGNORECASE)
                
                reasoning = reasoning_match.group(1).strip() if reasoning_match else f"Could not extract reasoning for step {i}"
                guideline = guideline_match.group(1).strip() if guideline_match else f"Could not extract guideline for step {i}"
                
                steps.append({
                    'step_number': i,
                    'reasoning': reasoning,
                    'hypothetical_guideline': guideline
                })
            else:
                steps.append({
                    'step_number': i,
                    'reasoning': f"Could not extract reasoning for step {i}",
                    'hypothetical_guideline': f"Could not extract guideline for step {i}"
                })
        
        return steps

    def _retrieve_for_reasoning_step(self, reasoning: str, guideline: str, original_question: str) -> str:
        """Retrieve relevant documents for a specific reasoning step and its guideline."""
        # Create a search query combining reasoning step, hypothetical guideline, and original question
        search_query = f"{reasoning} {guideline} {original_question}"
        
        try:
            # Use the RAG evaluator to retrieve documents
            retrieved_context = self.rag_evaluator.retrieve_context(
                search_query, 
                k=self.k_retrieval_per_guideline
            )
            return retrieved_context
        except Exception as e:
            print(f"    Error retrieving for reasoning step: {e}")
            return f"Error retrieving information for this reasoning step: {str(e)}"

    def _retrieve_unique_contexts_for_guidelines(self, guidelines: List[str], original_question: str) -> List[Dict]:
        """Retrieve unique document contexts for each guideline, ensuring no duplicates."""
        if not hasattr(self.rag_evaluator, 'vector_store') or not self.rag_evaluator.vector_store:
            print("Warning: Vector store not available for unique retrieval, using fallback method")
            # Fallback to original method
            guideline_retrievals = []
            for i, guideline in enumerate(guidelines, 1):
                retrieval = self._retrieve_for_guideline(guideline, original_question)
                guideline_retrievals.append({
                    'guideline': guideline,
                    'retrieved_context': retrieval
                })
            return guideline_retrievals
        
        # Get all candidate documents for all guidelines
        all_candidates = []
        guideline_queries = []
        
        for guideline in guidelines:
            search_query = f"{guideline} {original_question}"
            guideline_queries.append(search_query)
            
            # Get more candidates than needed to allow for unique selection
            retriever = self.rag_evaluator.vector_store.as_retriever(
                search_kwargs={"k": self.k_retrieval_per_guideline * len(guidelines) * 2}
            )
            documents = retriever.get_relevant_documents(search_query)
            
            # Sort deterministically by content hash for reproducible results
            documents = sorted(documents, key=lambda doc: hash(doc.page_content))
            
            all_candidates.append(documents)
        
        # Select unique documents for each guideline
        used_document_hashes = set()
        guideline_retrievals = []
        
        for i, (guideline, candidates) in enumerate(zip(guidelines, all_candidates)):
            selected_docs = []
            
            # Find the first k_retrieval_per_guideline unique documents for this guideline
            for doc in candidates:
                doc_hash = hash(doc.page_content)
                if doc_hash not in used_document_hashes and len(selected_docs) < self.k_retrieval_per_guideline:
                    selected_docs.append(doc)
                    used_document_hashes.add(doc_hash)
            
            # Format the retrieved context
            if selected_docs:
                context_parts = []
                for doc in selected_docs:
                    source_info = doc.metadata.get('source', 'Unknown source')
                    page_info = doc.metadata.get('page', 'N/A')
                    context_parts.append(f"Source: {source_info} (Page: {page_info})\n{doc.page_content}")
                retrieved_context = "\n\n---\n\n".join(context_parts)
            else:
                retrieved_context = f"No unique context found for guideline {i+1}"
            
            guideline_retrievals.append({
                'guideline': guideline,
                'retrieved_context': retrieved_context
            })
        
        return guideline_retrievals
    
    def _retrieve_for_guideline(self, guideline: str, original_question: str) -> str:
        """Retrieve relevant documents for a specific guideline."""
        # Create a search query that combines the guideline with the original question
        search_query = f"{guideline} {original_question}"
        
        try:
            # Use the RAG evaluator to retrieve documents
            retrieved_context = self.rag_evaluator.retrieve_context(
                search_query, 
                k=self.k_retrieval_per_guideline
            )
            return retrieved_context
        except Exception as e:
            print(f"    Error retrieving for guideline: {e}")
            return f"Error retrieving information for this guideline: {str(e)}"
    
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
        
        # Secondary fallback: Legacy patterns
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
    
    async def evaluate_sample_adaptive_async(self, 
                                           sample: Dict, 
                                           prompt_type: str = "direct") -> Dict:
        """Evaluate a single sample using adaptive RAG approach (async version)."""
        question = sample['Question']
        options = sample['Options']
        correct_answer = self.data_loader.get_correct_answer(sample)
        correct_choice = self.data_loader.get_answer_choice(sample)
        
        # Stage 1: Get medical guidelines from LLM
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        guideline_prompt = self.guideline_prompt_template.format(
            k=self.k_guidelines,
            question=question,
            options=options_text
        )
        
        stage1_start = time.time()
        guideline_response = await self._make_api_call_async(guideline_prompt)
        stage1_time = time.time() - stage1_start
        
        # Extract guidelines
        guidelines = self._extract_guidelines(guideline_response)
        
        # Stage 2: Retrieve information for each guideline (synchronous for now as RAG is sync)
        stage2_start = time.time()
        
        guideline_retrievals = self._retrieve_unique_contexts_for_guidelines(guidelines, question)
        
        stage2_time = time.time() - stage2_start
        
        # Stage 3: Answer the question using retrieved information
        # Combine all retrieved contexts WITHOUT the hypothetical guidelines
        # Only include the real retrieved information from the knowledge base
        combined_context = "\n\n" + "="*50 + "\n\n".join([
            f"RETRIEVED MEDICAL CONTEXT {i+1}:\n{item['retrieved_context']}"
            for i, item in enumerate(guideline_retrievals)
        ])
        
        # Use the appropriate RAG prompt type
        rag_prompt_type = f"{prompt_type}_rag" if "_rag" not in prompt_type else prompt_type
        
        try:
            final_prompt = PromptTemplates.get_prompt(rag_prompt_type, question, options, context=combined_context)
        except ValueError:
            # Fallback to basic RAG prompt if specific type not found
            final_prompt = PromptTemplates.get_prompt("direct_rag", question, options, context=combined_context)
            rag_prompt_type = "direct_rag"
        
        stage3_start = time.time()
        final_response = await self._make_api_call_async(final_prompt)
        stage3_time = time.time() - stage3_start
        
        # Extract predicted answer
        predicted_choice = self._extract_answer(final_response)
        is_correct = predicted_choice == correct_choice
        
        total_time = stage1_time + stage2_time + stage3_time
        
        return {
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'correct_choice': correct_choice,
            'prompt_type': f"adaptive_rag_{prompt_type}",
            'predicted_choice': predicted_choice,
            'is_correct': is_correct,
            'specialty': self.data_loader.get_sample_specialty(sample),
            
            # Adaptive RAG specific fields
            'adaptive_rag_used': True,
            'k_guidelines': self.k_guidelines,
            'k_retrieval_per_guideline': self.k_retrieval_per_guideline,
            
            # Stage 1: Guideline identification
            'stage1_guideline_prompt': guideline_prompt,
            'stage1_guideline_response': guideline_response,
            'stage1_identified_guidelines': guidelines,
            'stage1_time': stage1_time,
            
            # Stage 2: Information retrieval
            'stage2_guideline_retrievals': guideline_retrievals,
            'stage2_time': stage2_time,
            
            # Stage 3: Final answering
            'stage3_final_prompt': final_prompt,
            'stage3_final_response': final_response,
            'stage3_time': stage3_time,
            
            # Overall metrics
            'total_response_time': total_time,
            'total_api_calls_for_sample': 2,  # 1 for guidelines + 1 for final answer
        }
    
    def evaluate_sample_adaptive(self, 
                                sample: Dict, 
                                prompt_type: str = "direct") -> Dict:
        """Evaluate a single sample using adaptive RAG approach (sync version)."""
        question = sample['Question']
        options = sample['Options']
        correct_answer = self.data_loader.get_correct_answer(sample)
        correct_choice = self.data_loader.get_answer_choice(sample)
        
        print(f"\n🔍 Stage 1: Identifying medical guidelines needed...")
        
        # Stage 1: Get medical guidelines from LLM
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        guideline_prompt = self.guideline_prompt_template.format(
            k=self.k_guidelines,
            question=question,
            options=options_text
        )
        
        stage1_start = time.time()
        guideline_response = self._make_api_call(guideline_prompt)
        stage1_time = time.time() - stage1_start
        
        # Extract guidelines
        guidelines = self._extract_guidelines(guideline_response)
        print(f"  ✅ Identified {len(guidelines)} guidelines in {stage1_time:.2f}s")
        
        # Stage 2: Retrieve information for each guideline
        print(f"\n📚 Stage 2: Retrieving information for each guideline...")
        stage2_start = time.time()
        
        guideline_retrievals = self._retrieve_unique_contexts_for_guidelines(guidelines, question)
        
        stage2_time = time.time() - stage2_start
        print(f"  ✅ Retrieved information for all guidelines in {stage2_time:.2f}s")
        
        # Stage 3: Answer the question using retrieved information
        print(f"\n🧠 Stage 3: Answering question with targeted retrievals...")
        
        # Combine all retrieved contexts WITHOUT the hypothetical guidelines
        # Only include the real retrieved information from the knowledge base
        combined_context = "\n\n" + "="*50 + "\n\n".join([
            f"RETRIEVED MEDICAL CONTEXT {i+1}:\n{item['retrieved_context']}"
            for i, item in enumerate(guideline_retrievals)
        ])
        
        # Use the appropriate RAG prompt type
        rag_prompt_type = f"{prompt_type}_rag" if "_rag" not in prompt_type else prompt_type
        
        try:
            final_prompt = PromptTemplates.get_prompt(rag_prompt_type, question, options, context=combined_context)
        except ValueError:
            # Fallback to basic RAG prompt if specific type not found
            final_prompt = PromptTemplates.get_prompt("direct_rag", question, options, context=combined_context)
            rag_prompt_type = "direct_rag"
        
        stage3_start = time.time()
        final_response = self._make_api_call(final_prompt)
        stage3_time = time.time() - stage3_start
        
        # Extract predicted answer
        predicted_choice = self._extract_answer(final_response)
        is_correct = predicted_choice == correct_choice
        
        total_time = stage1_time + stage2_time + stage3_time
        
        print(f"  ✅ Final answer: {predicted_choice} ({'✓' if is_correct else '✗'}) in {stage3_time:.2f}s")
        print(f"  🕒 Total time: {total_time:.2f}s")
        
        return {
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'correct_choice': correct_choice,
            'prompt_type': f"adaptive_rag_{prompt_type}",
            'predicted_choice': predicted_choice,
            'is_correct': is_correct,
            'specialty': self.data_loader.get_sample_specialty(sample),
            
            # Adaptive RAG specific fields
            'adaptive_rag_used': True,
            'k_guidelines': self.k_guidelines,
            'k_retrieval_per_guideline': self.k_retrieval_per_guideline,
            
            # Stage 1: Guideline identification
            'stage1_guideline_prompt': guideline_prompt,
            'stage1_guideline_response': guideline_response,
            'stage1_identified_guidelines': guidelines,
            'stage1_time': stage1_time,
            
            # Stage 2: Information retrieval
            'stage2_guideline_retrievals': guideline_retrievals,
            'stage2_time': stage2_time,
            
            # Stage 3: Final answering
            'stage3_final_prompt': final_prompt,
            'stage3_final_response': final_response,
            'stage3_time': stage3_time,
            
            # Overall metrics
            'total_response_time': total_time,
            'total_api_calls_for_sample': 2,  # 1 for guidelines + 1 for final answer
        }
    
    async def evaluate_dataset_adaptive_concurrent(self, 
                                                  split: str = "test", 
                                                  prompt_types: List[str] = ["direct"],
                                                  sample_size: Optional[int] = None,
                                                  specialty_filter: Optional[str] = None,
                                                  save_results: bool = True) -> Dict:
        """Evaluate the model on a dataset split using adaptive RAG with concurrent processing."""
        
        print(f"🚀 Starting CONCURRENT ADAPTIVE RAG evaluation on {split} split")
        print("=" * 70)
        print(f"🎯 Deterministic mode: ENABLED (seed={RANDOM_SEED})")
        print(f"🌡️  Temperature: {self.model_config['temperature']} (deterministic)")
        print(f"Model: {self.model_name}")
        print(f"Prompt types: {prompt_types}")
        print(f"Guidelines per question: {self.k_guidelines}")
        print(f"Retrievals per guideline: {self.k_retrieval_per_guideline}")
        print(f"Max concurrent requests: {self.max_concurrent}")
        print(f"Requests per minute: {self.requests_per_minute}")
        
        # Get data and sort deterministically
        data = self.data_loader.get_split(split, sample_size, specialty_filter)
        # Sort by question text for consistent ordering across runs
        data = sorted(data, key=lambda x: x['Question'])  
        print(f"Evaluating on {len(data)} samples (deterministically sorted)")
        
        all_results = []
        
        for prompt_type in prompt_types:
            print(f"\n🧪 Evaluating with {prompt_type} + adaptive RAG (CONCURRENT)...")
            
            # Create tasks for concurrent processing
            tasks = []
            for sample in data:
                task = self.evaluate_sample_adaptive_async(sample, prompt_type)
                tasks.append(task)
            
            # Execute tasks concurrently with progress bar
            prompt_results = []
            async for result in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Concurrent Adaptive RAG ({prompt_type})"):
                try:
                    completed_result = await result
                    prompt_results.append(completed_result)
                    all_results.append(completed_result)
                except Exception as e:
                    print(f"❌ Error processing sample: {e}")
                    continue
            
            # Calculate accuracy for this prompt type
            correct_count = sum(1 for r in prompt_results if r['is_correct'])
            accuracy = correct_count / len(prompt_results) if prompt_results else 0
            print(f"\n📊 {prompt_type} + adaptive RAG accuracy: {accuracy:.3f} ({correct_count}/{len(prompt_results)})")
        
        # Generate comprehensive results
        results_summary = self._generate_results_summary(all_results, split)
        
        if save_results:
            self._save_results(all_results, results_summary, split, concurrent=True)
        
        return {
            'detailed_results': all_results,
            'summary': results_summary,
            'api_usage': {
                'total_calls': self.api_calls,
                'total_tokens': self.total_tokens,
                'adaptive_rag_processing': True,
                'concurrent_processing': True,
                'max_concurrent_requests': self.max_concurrent,
                'requests_per_minute': self.requests_per_minute,
                'deterministic_mode': True,
                'random_seed': RANDOM_SEED
            }
        }
    
    def evaluate_dataset_adaptive(self, 
                                 split: str = "test", 
                                 prompt_types: List[str] = ["direct"],
                                 sample_size: Optional[int] = None,
                                 specialty_filter: Optional[str] = None,
                                 save_results: bool = True,
                                 concurrent: bool = False) -> Dict:
        """Evaluate the model on a dataset split using adaptive RAG."""
        
        if concurrent:
            # Use async concurrent version
            return asyncio.run(self.evaluate_dataset_adaptive_concurrent(
                split, prompt_types, sample_size, specialty_filter, save_results
            ))
        
        # Use synchronous version (existing implementation)
        print(f"🚀 Starting ADAPTIVE RAG evaluation on {split} split")
        print("=" * 60)
        print(f"🎯 Deterministic mode: ENABLED (seed={RANDOM_SEED})")
        print(f"🌡️  Temperature: {self.model_config['temperature']} (deterministic)")
        print(f"Model: {self.model_name}")
        print(f"Prompt types: {prompt_types}")
        print(f"Guidelines per question: {self.k_guidelines}")
        print(f"Retrievals per guideline: {self.k_retrieval_per_guideline}")
        
        # Get data and sort deterministically
        data = self.data_loader.get_split(split, sample_size, specialty_filter)
        # Sort by question text for consistent ordering across runs
        data = sorted(data, key=lambda x: x['Question'])  
        print(f"Evaluating on {len(data)} samples (deterministically sorted)")
        
        all_results = []
        
        for prompt_type in prompt_types:
            print(f"\n🧪 Evaluating with {prompt_type} + adaptive RAG...")
            
            prompt_results = []
            for i, sample in enumerate(tqdm(data, desc=f"Adaptive RAG ({prompt_type})")):
                try:
                    print(f"\n📋 Sample {i+1}/{len(data)}")
                    result = self.evaluate_sample_adaptive(sample, prompt_type)
                    prompt_results.append(result)
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Error processing sample {i+1}: {e}")
                    continue
            
            # Calculate accuracy for this prompt type
            correct_count = sum(1 for r in prompt_results if r['is_correct'])
            accuracy = correct_count / len(prompt_results) if prompt_results else 0
            print(f"\n📊 {prompt_type} + adaptive RAG accuracy: {accuracy:.3f} ({correct_count}/{len(prompt_results)})")
        
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
                'adaptive_rag_processing': True,
                'deterministic_mode': True,
                'random_seed': RANDOM_SEED
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
                'adaptive_rag_processing': True,
                'k_guidelines': self.k_guidelines,
                'k_retrieval_per_guideline': self.k_retrieval_per_guideline,
                'max_concurrent': self.max_concurrent,
                'requests_per_minute': self.requests_per_minute,
                # Deterministic settings
                'deterministic_mode': True,
                'random_seed': RANDOM_SEED,
                'temperature': self.model_config['temperature'],
                'top_p': self.model_config['top_p'],
                'frequency_penalty': self.model_config['frequency_penalty'],
                'presence_penalty': self.model_config['presence_penalty'],
                'api_seed': self.model_config['seed']
            },
            'overall_performance': {},
            'performance_by_prompt': {},
            'performance_by_specialty': {},
            'adaptive_rag_metrics': {},
            'error_analysis': {}
        }
        
        # Overall performance
        overall_accuracy = float(df['is_correct'].mean())
        summary['overall_performance']['accuracy'] = overall_accuracy
        summary['overall_performance']['correct_count'] = int(df['is_correct'].sum())
        summary['overall_performance']['total_count'] = len(df)
        summary['overall_performance']['avg_total_time'] = float(df['total_response_time'].mean())
        
        # Performance by prompt type
        for prompt_type in df['prompt_type'].unique():
            prompt_df = df[df['prompt_type'] == prompt_type]
            summary['performance_by_prompt'][prompt_type] = {
                'accuracy': float(prompt_df['is_correct'].mean()),
                'correct_count': int(prompt_df['is_correct'].sum()),
                'total_count': len(prompt_df),
                'avg_total_time': float(prompt_df['total_response_time'].mean())
            }
        
        # Performance by specialty
        for specialty in df['specialty'].unique():
            specialty_df = df[df['specialty'] == specialty]
            summary['performance_by_specialty'][specialty] = {
                'accuracy': float(specialty_df['is_correct'].mean()),
                'correct_count': int(specialty_df['is_correct'].sum()),
                'total_count': len(specialty_df)
            }
        
        # Adaptive RAG specific metrics
        summary['adaptive_rag_metrics'] = {
            'avg_stage1_time': float(df['stage1_time'].mean()),
            'avg_stage2_time': float(df['stage2_time'].mean()),
            'avg_stage3_time': float(df['stage3_time'].mean()),
            'avg_total_time': float(df['total_response_time'].mean()),
            'avg_api_calls_per_sample': float(df['total_api_calls_for_sample'].mean())
        }
        
        # Error analysis
        incorrect_df = df[~df['is_correct']]
        summary['error_analysis']['total_errors'] = len(incorrect_df)
        summary['error_analysis']['unknown_answers'] = int((df['predicted_choice'] == 'UNKNOWN').sum())
        
        return summary
    
    def _save_results(self, results: List[Dict], summary: Dict, split: str, concurrent: bool = False) -> None:
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_concurrent" if concurrent else ""
        
        # Create results directory if it doesn't exist
        import os
        os.makedirs("results", exist_ok=True)
        
        # Save detailed results
        detailed_file = f"results/{self.model_name}_{split}_{timestamp}_adaptive_rag{mode_suffix}_detailed.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        summary_file = f"results/{self.model_name}_{split}_{timestamp}_adaptive_rag{mode_suffix}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save CSV for easy analysis
        csv_file = f"results/{self.model_name}_{split}_{timestamp}_adaptive_rag{mode_suffix}.csv"
        pd.DataFrame(results).to_csv(csv_file, index=False)
        
        print(f"\n📁 Results saved:")
        print(f"  Detailed: {detailed_file}")
        print(f"  Summary: {summary_file}")
        print(f"  CSV: {csv_file}")

    def _retrieve_unique_contexts_for_reasoning_steps(self, reasoning_steps: List[Dict], original_question: str) -> List[Dict]:
        """Retrieve unique document contexts for each reasoning step, ensuring no duplicates."""
        if not hasattr(self.rag_evaluator, 'vector_store') or not self.rag_evaluator.vector_store:
            print("Warning: Vector store not available for unique retrieval, using fallback method")
            # Fallback to original method
            step_retrievals = []
            for step in reasoning_steps:
                retrieval = self._retrieve_for_reasoning_step(
                    step['reasoning'], 
                    step['hypothetical_guideline'], 
                    original_question
                )
                step_retrievals.append({
                    **step,
                    'retrieved_context': retrieval
                })
            return step_retrievals
        
        # Get all candidate documents for all reasoning steps
        all_candidates = []
        step_queries = []
        
        for step in reasoning_steps:
            search_query = f"{step['reasoning']} {step['hypothetical_guideline']} {original_question}"
            step_queries.append(search_query)
            
            # Get more candidates than needed to allow for unique selection
            retriever = self.rag_evaluator.vector_store.as_retriever(
                search_kwargs={"k": self.k_retrieval_per_guideline * len(reasoning_steps) * 2}
            )
            documents = retriever.get_relevant_documents(search_query)
            
            # Sort deterministically by content hash for reproducible results
            documents = sorted(documents, key=lambda doc: hash(doc.page_content))
            
            all_candidates.append(documents)
        
        # Select unique documents for each reasoning step
        used_document_hashes = set()
        step_retrievals = []
        
        for i, (step, candidates) in enumerate(zip(reasoning_steps, all_candidates)):
            selected_docs = []
            
            # Find the first k_retrieval_per_guideline unique documents for this step
            for doc in candidates:
                doc_hash = hash(doc.page_content)
                if doc_hash not in used_document_hashes and len(selected_docs) < self.k_retrieval_per_guideline:
                    selected_docs.append(doc)
                    used_document_hashes.add(doc_hash)
            
            # Format the retrieved context
            if selected_docs:
                context_parts = []
                for doc in selected_docs:
                    source_info = doc.metadata.get('source', 'Unknown source')
                    page_info = doc.metadata.get('page', 'N/A')
                    context_parts.append(f"Source: {source_info} (Page: {page_info})\n{doc.page_content}")
                retrieved_context = "\n\n---\n\n".join(context_parts)
            else:
                retrieved_context = f"No unique context found for reasoning step {step['step_number']}"
            
            step_retrievals.append({
                **step,
                'retrieved_context': retrieved_context
            })
        
        return step_retrievals

    async def evaluate_sample_stepwise_guideline_async(self, 
                                                      sample: Dict, 
                                                      prompt_type: str = "direct") -> Dict:
        """Evaluate a single sample using step-by-step guideline RAG approach (async version)."""
        question = sample['Question']
        options = sample['Options']
        correct_answer = self.data_loader.get_correct_answer(sample)
        correct_choice = self.data_loader.get_answer_choice(sample)
        
        # Stage 1: Get step-by-step reasoning with hypothetical guidelines from LLM
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        stepwise_prompt = self.stepwise_guideline_prompt_template.format(
            k=self.k_guidelines,
            question=question,
            options=options_text
        )
        
        stage1_start = time.time()
        stepwise_response = await self._make_api_call_async(stepwise_prompt)
        stage1_time = time.time() - stage1_start
        
        # Extract reasoning steps and hypothetical guidelines
        reasoning_steps = self._extract_reasoning_steps(stepwise_response)
        
        # Stage 2: Retrieve real information for each reasoning step
        stage2_start = time.time()
        step_retrievals = self._retrieve_unique_contexts_for_reasoning_steps(reasoning_steps, question)
        stage2_time = time.time() - stage2_start
        
        # Stage 3: Refine reasoning with real guidelines and provide final answer
        # Combine reasoning steps with real retrieved contexts
        refined_context = "\n\n" + "="*50 + "\n\n".join([
            f"REASONING STEP {item['step_number']}:\n{item['reasoning']}\n\nREAL MEDICAL GUIDELINE CONTEXT:\n{item['retrieved_context']}"
            for item in step_retrievals
        ])
        
        # Create refinement prompt
        refinement_prompt = f"""You are a medical expert. You previously reasoned through this clinical question step by step, but now you have access to real medical guideline excerpts for each reasoning step.

Please refine your reasoning based on the real medical guidelines provided and give your final answer.

Original Question: {question}

Options:
{options_text}

Your Previous Reasoning Steps with Real Medical Guidelines:
{refined_context}

Based on the real medical guidelines provided above, please refine your step-by-step reasoning and provide your final answer.

Please provide your answer in the following format:
<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""
        
        stage3_start = time.time()
        final_response = await self._make_api_call_async(refinement_prompt)
        stage3_time = time.time() - stage3_start
        
        # Extract predicted answer
        predicted_choice = self._extract_answer(final_response)
        is_correct = predicted_choice == correct_choice
        
        total_time = stage1_time + stage2_time + stage3_time
        
        return {
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'correct_choice': correct_choice,
            'prompt_type': f"stepwise_guideline_rag_{prompt_type}",
            'predicted_choice': predicted_choice,
            'is_correct': is_correct,
            'specialty': self.data_loader.get_sample_specialty(sample),
            
            # Step-by-step Guideline RAG specific fields
            'stepwise_guideline_rag_used': True,
            'k_guidelines': self.k_guidelines,
            'k_retrieval_per_guideline': self.k_retrieval_per_guideline,
            
            # Stage 1: Step-by-step reasoning with hypothetical guidelines
            'stage1_stepwise_prompt': stepwise_prompt,
            'stage1_stepwise_response': stepwise_response,
            'stage1_reasoning_steps': reasoning_steps,
            'stage1_time': stage1_time,
            
            # Stage 2: Real information retrieval
            'stage2_step_retrievals': step_retrievals,
            'stage2_time': stage2_time,
            
            # Stage 3: Refined reasoning with real guidelines
            'stage3_refinement_prompt': refinement_prompt,
            'stage3_final_response': final_response,
            'stage3_time': stage3_time,
            
            # Overall metrics
            'total_response_time': total_time,
            'total_api_calls_for_sample': 2,  # 1 for initial reasoning + 1 for refinement
        }

    def evaluate_sample_stepwise_guideline(self, 
                                          sample: Dict, 
                                          prompt_type: str = "direct") -> Dict:
        """Evaluate a single sample using step-by-step guideline RAG approach (sync version)."""
        question = sample['Question']
        options = sample['Options']
        correct_answer = self.data_loader.get_correct_answer(sample)
        correct_choice = self.data_loader.get_answer_choice(sample)
        
        print(f"\n🧠 Stage 1: Generating step-by-step reasoning with hypothetical guidelines...")
        
        # Stage 1: Get step-by-step reasoning with hypothetical guidelines from LLM
        options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        stepwise_prompt = self.stepwise_guideline_prompt_template.format(
            k=self.k_guidelines,
            question=question,
            options=options_text
        )
        
        stage1_start = time.time()
        stepwise_response = self._make_api_call(stepwise_prompt)
        stage1_time = time.time() - stage1_start
        
        # Extract reasoning steps and hypothetical guidelines
        reasoning_steps = self._extract_reasoning_steps(stepwise_response)
        print(f"  ✅ Generated {len(reasoning_steps)} reasoning steps in {stage1_time:.2f}s")
        
        # Stage 2: Retrieve real information for each reasoning step
        print(f"\n📚 Stage 2: Retrieving real medical guidelines for each reasoning step...")
        stage2_start = time.time()
        step_retrievals = self._retrieve_unique_contexts_for_reasoning_steps(reasoning_steps, question)
        stage2_time = time.time() - stage2_start
        print(f"  ✅ Retrieved real guidelines for all steps in {stage2_time:.2f}s")
        
        # Stage 3: Refine reasoning with real guidelines and provide final answer
        print(f"\n🔬 Stage 3: Refining reasoning with real medical guidelines...")
        
        # Combine reasoning steps with real retrieved contexts
        refined_context = "\n\n" + "="*50 + "\n\n".join([
            f"REASONING STEP {item['step_number']}:\n{item['reasoning']}\n\nREAL MEDICAL GUIDELINE CONTEXT:\n{item['retrieved_context']}"
            for item in step_retrievals
        ])
        
        # Create refinement prompt
        refinement_prompt = f"""You are a medical expert. You previously reasoned through this clinical question step by step, but now you have access to real medical guideline excerpts for each reasoning step.

Please refine your reasoning based on the real medical guidelines provided and give your final answer.

Original Question: {question}

Options:
{options_text}

Your Previous Reasoning Steps with Real Medical Guidelines:
{refined_context}

Based on the real medical guidelines provided above, please refine your step-by-step reasoning and provide your final answer.

Please provide your answer in the following format:
<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""
        
        stage3_start = time.time()
        final_response = self._make_api_call(refinement_prompt)
        stage3_time = time.time() - stage3_start
        
        # Extract predicted answer
        predicted_choice = self._extract_answer(final_response)
        is_correct = predicted_choice == correct_choice
        
        total_time = stage1_time + stage2_time + stage3_time
        
        print(f"  ✅ Final answer: {predicted_choice} ({'✓' if is_correct else '✗'}) in {stage3_time:.2f}s")
        print(f"  🕒 Total time: {total_time:.2f}s")
        
        return {
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'correct_choice': correct_choice,
            'prompt_type': f"stepwise_guideline_rag_{prompt_type}",
            'predicted_choice': predicted_choice,
            'is_correct': is_correct,
            'specialty': self.data_loader.get_sample_specialty(sample),
            
            # Step-by-step Guideline RAG specific fields
            'stepwise_guideline_rag_used': True,
            'k_guidelines': self.k_guidelines,
            'k_retrieval_per_guideline': self.k_retrieval_per_guideline,
            
            # Stage 1: Step-by-step reasoning with hypothetical guidelines
            'stage1_stepwise_prompt': stepwise_prompt,
            'stage1_stepwise_response': stepwise_response,
            'stage1_reasoning_steps': reasoning_steps,
            'stage1_time': stage1_time,
            
            # Stage 2: Real information retrieval
            'stage2_step_retrievals': step_retrievals,
            'stage2_time': stage2_time,
            
            # Stage 3: Refined reasoning with real guidelines
            'stage3_refinement_prompt': refinement_prompt,
            'stage3_final_response': final_response,
            'stage3_time': stage3_time,
            
            # Overall metrics
            'total_response_time': total_time,
            'total_api_calls_for_sample': 2,  # 1 for initial reasoning + 1 for refinement
        }

# Example usage
if __name__ == "__main__":
    # Initialize adaptive RAG evaluator with concurrent support
    evaluator = AdaptiveRAGEvaluator(
        model_name="gpt-4o-mini",
        k_guidelines=3,
        k_retrieval_per_guideline=2,
        max_concurrent=10,
        requests_per_minute=100
    )
    
    # Run concurrent adaptive RAG evaluation
    print("Running concurrent adaptive RAG evaluation test...")
    results = evaluator.evaluate_dataset_adaptive(
        split="test",
        prompt_types=["direct", "chain_of_thought"],
        sample_size=3,  # Small sample for testing
        specialty_filter="Cardiology",
        save_results=True,
        concurrent=True  # Enable concurrent processing
    )
    
    print(f"\n🎉 Concurrent Adaptive RAG evaluation completed!")
    print(f"Overall accuracy: {results['summary']['overall_performance']['accuracy']:.3f}")
    print(f"API calls made: {results['api_usage']['total_calls']}")
    print(f"Average time per sample: {results['summary']['adaptive_rag_metrics']['avg_total_time']:.1f}s") 