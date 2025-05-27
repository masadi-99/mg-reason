#!/usr/bin/env python3
"""
Microsoft GraphRAG implementation for cardiology knowledge evaluation.
Uses gpt-4o-mini for cost efficiency.
"""

import os
import asyncio
import json
import pandas as pd
import time
from typing import List, Dict, Optional
from pathlib import Path
import tempfile
import shutil

from config import OPENAI_API_KEY

class GraphRAGEvaluator:
    """Microsoft GraphRAG-based system for cardiology knowledge evaluation."""
    
    def __init__(self, 
                 model_name: str = "gpt-4o-mini",
                 work_dir: str = "./graphrag_workspace"):
        """Initialize the GraphRAG evaluator."""
        
        self.model_name = model_name
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # Set up directories
        self.input_dir = self.work_dir / "input"
        self.output_dir = self.work_dir / "output"
        self.input_dir.mkdir(exist_ok=True)
        
        # Set environment variables
        os.environ["GRAPHRAG_API_KEY"] = OPENAI_API_KEY
        
        print(f"🧠 GraphRAG initialized with model: {model_name}")
        print(f"📁 Working directory: {work_dir}")
    
    async def _check_graphrag_installation(self):
        """Check if graphrag is installed and offer to install it."""
        try:
            import graphrag
            print("✅ GraphRAG installation found.")
        except ImportError:
            print("❌ GraphRAG is not installed.")
            # Note: We can't directly prompt user for installation here.
            # The user should ensure graphrag is installed as per requirements.txt.
            print("Please install GraphRAG by running: pip install graphrag")
            raise ImportError("GraphRAG is not installed. Please run 'pip install graphrag'")

    async def prepare_documents(self, pdf_directory: str = "./cardiology_pdfs") -> int:
        """Convert PDFs to text files for GraphRAG processing."""
        await self._check_graphrag_installation()
        print(f"📚 Preparing documents from {pdf_directory}...")
        
        # Clear previous input files
        for txt_file in self.input_dir.glob("*.txt"):
            txt_file.unlink()
        
        # Import PyPDF2 for text extraction
        try:
            from pypdf import PdfReader
        except ImportError:
            print("❌ pypdf not available. Installing...")
            os.system("pip install pypdf")
            from pypdf import PdfReader
        
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        processed_count = 0
        
        for pdf_file in pdf_files:
            try:
                print(f"  Processing: {pdf_file.name}")
                
                # Extract text from PDF
                reader = PdfReader(str(pdf_file))
                text_content = []
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(f"=== Page {page_num + 1} ===\n{text}\n")
                
                # Save as text file
                if text_content:
                    output_file = self.input_dir / f"{pdf_file.stem}.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"# {pdf_file.stem.replace('_', ' ')}\n\n")
                        f.write("".join(text_content))
                    
                    processed_count += 1
                    print(f"    ✅ Extracted {len(text_content)} pages")
                else:
                    print(f"    ⚠️ No text extracted")
                    
            except Exception as e:
                print(f"    ❌ Error processing {pdf_file.name}: {e}")
                continue
        
        print(f"📄 Prepared {processed_count} text files from {len(pdf_files)} PDFs")
        return processed_count
    
    async def create_settings_yaml(self):
        """Create GraphRAG settings.yaml configuration file."""
        await self._check_graphrag_installation()
        settings_content = f"""
encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: ${{GRAPHRAG_API_KEY}}
  type: openai_chat
  model: {self.model_name}
  model_supports_json: true
  max_tokens: 4000
  temperature: 0
  top_p: 1
  request_timeout: 180.0
  api_base: null
  api_version: null
  organization: null
  proxy: null
  cognitive_services_endpoint: null
  deployment_name: null
  tokens_per_minute: 0
  requests_per_minute: 0
  max_retries: 10
  max_retry_wait: 10.0
  sleep_on_rate_limit_recommendation: true
  concurrent_requests: 5

parallelization:
  stagger: 0.3
  num_threads: 20

async_mode: threaded

embeddings:
  async_mode: threaded
  llm:
    api_key: ${{GRAPHRAG_API_KEY}}
    type: openai_embedding
    model: text-embedding-ada-002
    api_base: null
    api_version: null
    organization: null
    proxy: null
    cognitive_services_endpoint: null
    deployment_name: null
    tokens_per_minute: 0
    requests_per_minute: 0
    max_retries: 10
    max_retry_wait: 10.0
    sleep_on_rate_limit_recommendation: true
    concurrent_requests: 5
  parallelization:
    stagger: 0.3
    num_threads: 20
  batch_size: 16
  batch_max_tokens: 8191
  target: required
  strategy:
    type: openai

input:
  type: file
  file_type: text
  base_dir: "input"
  file_encoding: utf-8
  file_pattern: ".*\\.txt$"

cache:
  type: file
  base_dir: "cache"

storage:
  type: file
  base_dir: "output"

update_index_storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "output"

entity_extraction:
  llm:
    api_key: ${{GRAPHRAG_API_KEY}}
    type: openai_chat
    model: {self.model_name}
    model_supports_json: true
    max_tokens: 4000
    temperature: 0
    top_p: 1
    request_timeout: 180.0
    api_base: null
    api_version: null
    organization: null
    proxy: null
    cognitive_services_endpoint: null
    deployment_name: null
    tokens_per_minute: 0
    requests_per_minute: 0
    max_retries: 10
    max_retry_wait: 10.0
    sleep_on_rate_limit_recommendation: true
    concurrent_requests: 5
  parallelization:
    stagger: 0.3
    num_threads: 20
  async_mode: threaded
  prompt: "prompts/entity_extraction.txt"
  entity_types: [person, organization, geo, event, medication, condition, procedure, symptom]
  max_gleanings: 1

summarize_descriptions:
  llm:
    api_key: ${{GRAPHRAG_API_KEY}}
    type: openai_chat
    model: {self.model_name}
    model_supports_json: true
    max_tokens: 4000
    temperature: 0
    top_p: 1
    request_timeout: 180.0
    api_base: null
    api_version: null
    organization: null
    proxy: null
    cognitive_services_endpoint: null
    deployment_name: null
    tokens_per_minute: 0
    requests_per_minute: 0
    max_retries: 10
    max_retry_wait: 10.0
    sleep_on_rate_limit_recommendation: true
    concurrent_requests: 5
  parallelization:
    stagger: 0.3
    num_threads: 20
  async_mode: threaded
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

claim_extraction:
  llm:
    api_key: ${{GRAPHRAG_API_KEY}}
    type: openai_chat
    model: {self.model_name}
    model_supports_json: true
    max_tokens: 4000
    temperature: 0
    top_p: 1
    request_timeout: 180.0
    api_base: null
    api_version: null
    organization: null
    proxy: null
    cognitive_services_endpoint: null
    deployment_name: null
    tokens_per_minute: 0
    requests_per_minute: 0
    max_retries: 10
    max_retry_wait: 10.0
    sleep_on_rate_limit_recommendation: true
    concurrent_requests: 5
  parallelization:
    stagger: 0.3
    num_threads: 20
  async_mode: threaded
  prompt: "prompts/claim_extraction.txt"
  description: "Any claims or facts that could be relevant to medical decision making."
  max_gleanings: 1

community_report:
  llm:
    api_key: ${{GRAPHRAG_API_KEY}}
    type: openai_chat
    model: {self.model_name}
    model_supports_json: true
    max_tokens: 4000
    temperature: 0
    top_p: 1
    request_timeout: 180.0
    api_base: null
    api_version: null
    organization: null
    proxy: null
    cognitive_services_endpoint: null
    deployment_name: null
    tokens_per_minute: 0
    requests_per_minute: 0
    max_retries: 10
    max_retry_wait: 10.0
    sleep_on_rate_limit_recommendation: true
    concurrent_requests: 5
  parallelization:
    stagger: 0.3
    num_threads: 20
  async_mode: threaded
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

text_unit:
  size: 1500
  overlap: 150

embed_graph:
  enabled: false

umap:
  enabled: false

snapshots:
  graphml: false
  raw_entities: false
  top_level_nodes: false

local_search:
  text_unit_prop: 0.5
  community_prop: 0.1
  conversation_history_max_turns: 5
  top_k_mapped_entities: 10
  top_k_relationships: 10
  max_tokens: 12000

global_search:
  max_tokens: 12000
  data_max_tokens: 12000
  map_max_tokens: 1000
  reduce_max_tokens: 2000
  concurrency: 32
"""
        
        settings_file = self.work_dir / "settings.yaml"
        with open(settings_file, 'w') as f:
            f.write(settings_content)
        
        print(f"⚙️ Created settings.yaml at {settings_file}")
    
    async def run_indexing(self) -> bool:
        """Run GraphRAG indexing pipeline."""
        await self._check_graphrag_installation()
        print("🏗️ Running GraphRAG indexing...")
        
        # Change to work directory
        original_cwd = os.getcwd()
        os.chdir(self.work_dir)
        
        try:
            # Run graphrag index command
            start_time = time.time()
            exit_code = os.system("graphrag index --root .")
            elapsed_time = time.time() - start_time
            
            if exit_code == 0:
                print(f"✅ Indexing completed successfully in {elapsed_time:.2f} seconds")
                return True
            else:
                print(f"❌ Indexing failed with exit code: {exit_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error during indexing: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    async def query_global(self, question: str, k: int = 5) -> Dict:
        """Query using GraphRAG global search."""
        await self._check_graphrag_installation()
        print(f"🌍 Global query: {question}")
        
        original_cwd = os.getcwd()
        os.chdir(self.work_dir)
        
        try:
            start_time = time.time()
            
            # Run global search command and capture output
            import subprocess
            result = subprocess.run(
                ["graphrag", "query", "--root", ".", "--method", "global", "--query", question],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            query_time = time.time() - start_time
            
            if result.returncode == 0:
                answer = result.stdout.strip()
                print(f"✅ Global query completed in {query_time:.2f} seconds")
                
                return {
                    "answer": answer,
                    "method": "graphrag_global",
                    "model": self.model_name,
                    "query_time": query_time,
                    "question": question
                }
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                print(f"❌ Global query failed: {error_msg}")
                return {
                    "answer": f"Error: {error_msg}",
                    "method": "graphrag_global",
                    "model": self.model_name,
                    "query_time": query_time,
                    "question": question
                }
                
        except Exception as e:
            print(f"❌ Error during global query: {e}")
            return {
                "answer": f"Error: {e}",
                "method": "graphrag_global",
                "model": self.model_name,
                "query_time": 0,
                "question": question
            }
        finally:
            os.chdir(original_cwd)
    
    async def query_local(self, question: str, community_level: int = 2, k: int = 5) -> Dict:
        """Query using GraphRAG local search."""
        await self._check_graphrag_installation()
        print(f"🎯 Local query: {question}")
        
        original_cwd = os.getcwd()
        os.chdir(self.work_dir)
        
        try:
            start_time = time.time()
            
            # Run local search command and capture output
            import subprocess
            result = subprocess.run(
                ["graphrag", "query", "--root", ".", "--method", "local", "--query", question],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            query_time = time.time() - start_time
            
            if result.returncode == 0:
                answer = result.stdout.strip()
                print(f"✅ Local query completed in {query_time:.2f} seconds")
                
                return {
                    "answer": answer,
                    "method": "graphrag_local",
                    "model": self.model_name,
                    "query_time": query_time,
                    "question": question
                }
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                print(f"❌ Local query failed: {error_msg}")
                return {
                    "answer": f"Error: {error_msg}",
                    "method": "graphrag_local",
                    "model": self.model_name,
                    "query_time": query_time,
                    "question": question
                }
                
        except Exception as e:
            print(f"❌ Error during local query: {e}")
            return {
                "answer": f"Error: {e}",
                "method": "graphrag_local",
                "model": self.model_name,
                "query_time": 0,
                "question": question
            }
        finally:
            os.chdir(original_cwd)
    
    async def retrieve_context(self, question: str, search_type: str = "global", k: int = 5, community_level: int = 2) -> str:
        """Retrieve relevant context for a given question using GraphRAG.
        
        Args:
            question: The question to retrieve context for.
            search_type: 'global' or 'local'. Defaults to 'global'.
            k: Number of documents/context pieces to retrieve.
            community_level: Community level for local search.
        
        Returns:
            A string containing the formatted retrieved context.
        """
        await self._check_graphrag_installation()
        print(f"🔍 Retrieving context for GraphRAG ({search_type} search): {question}")
        start_time = time.time()

        if not (self.output_dir / "artifacts").exists():
            print("GraphRAG artifacts not found. Running indexing first...")
            if not self.input_dir.exists() or not any(self.input_dir.iterdir()):
                await self.prepare_documents()
            if not (self.work_dir / "settings.yaml").exists():
                await self.create_settings_yaml()
            await self.run_indexing()
            if not (self.output_dir / "artifacts").exists():
                raise FileNotFoundError("GraphRAG indexing failed or did not produce artifacts.")
        
        from graphrag.query.cli import run_global_search, run_local_search
        from graphrag.config.enums import InputFileType

        # Create a temporary directory for this specific query if needed by GraphRAG CLI
        # For simplicity, we're using the main work_dir
        
        response_data = "No relevant context found."
        documents_info = []

        if search_type == "global":
            # For global search, we might need to parse the response to get structured sources
            # The `run_global_search` directly prints. We might need to capture stdout or adapt.
            # This is a simplification; true context extraction might need deeper integration.
            # For now, we will assume the answer itself contains enough context or we rely on a full query response.
            # Let's try to get the response from the query function to extract sources.
            query_result = await self.query_global(question, k=k)
            response_data = query_result["answer"]
            if query_result.get("sources"): # query_global should populate sources if possible
                for src in query_result["sources"]:
                    documents_info.append(f"Source: {src.get('document_id', 'Unknown')} (Confidence: {src.get('distance', 'N/A')})\\n{src.get('text', '')}")
            elif response_data:
                documents_info.append(f"Global Search Response:\\n{response_data}")

        elif search_type == "local":
            query_result = await self.query_local(question, community_level=community_level, k=k)
            response_data = query_result["answer"]
            if query_result.get("sources"): 
                for src in query_result["sources"]:
                    documents_info.append(f"Source: {src.get('document_id', 'Unknown')} (Level: {src.get('level', 'N/A')}, Page: {src.get('page', 'N/A')})\\n{src.get('text', '')}")
            elif response_data:
                documents_info.append(f"Local Search Response (Community {community_level}):\\n{response_data}")
        else:
            raise ValueError("Invalid search_type. Must be 'global' or 'local'.")

        end_time = time.time()
        print(f"  GraphRAG context retrieval ({search_type}) completed in {end_time - start_time:.2f} seconds.")

        if not documents_info:
            return "No relevant context found in the GraphRAG knowledge base for this question."
        
        return "\\n\\n---\\n\\n".join(documents_info)

    async def evaluate_on_dataset(self, questions: List[str], use_global: bool = True) -> List[Dict]:
        """Evaluate GraphRAG on a list of questions."""
        await self._check_graphrag_installation()
        method = "global" if use_global else "local"
        print(f"📊 Evaluating GraphRAG ({method}) on {len(questions)} questions...")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n--- Question {i}/{len(questions)} ---")
            
            try:
                if use_global:
                    result = await self.query_global(question)
                else:
                    result = await self.query_local(question)
                
                results.append(result)
                
            except Exception as e:
                print(f"❌ Error processing question {i}: {e}")
                results.append({
                    "question": question,
                    "answer": f"Error: {e}",
                    "method": f"graphrag_{method}",
                    "model": self.model_name,
                    "query_time": 0
                })
        
        print(f"\n✅ GraphRAG evaluation completed: {len(results)} results")
        return results
    
    async def get_index_stats(self) -> Dict:
        """Get statistics about the GraphRAG index."""
        await self._check_graphrag_installation()
        try:
            # Check if output directory exists and has files
            if not self.output_dir.exists():
                return {"error": "No index found. Run indexing first."}
            
            artifacts = list(self.output_dir.glob("artifacts/*.parquet"))
            
            stats = {
                "work_dir": str(self.work_dir),
                "input_files": len(list(self.input_dir.glob("*.txt"))),
                "artifact_files": len(artifacts),
                "model": self.model_name,
                "indexing_complete": len(artifacts) > 0
            }
            
            # Try to read some basic stats from artifacts
            if artifacts:
                try:
                    # Check for entities
                    entity_files = [f for f in artifacts if "entities" in f.name]
                    if entity_files:
                        entities_df = pd.read_parquet(entity_files[0])
                        stats["total_entities"] = len(entities_df)
                    
                    # Check for relationships
                    relationship_files = [f for f in artifacts if "relationships" in f.name]
                    if relationship_files:
                        relationships_df = pd.read_parquet(relationship_files[0])
                        stats["total_relationships"] = len(relationships_df)
                        
                except Exception as e:
                    stats["stats_error"] = str(e)
            
            return stats
            
        except Exception as e:
            return {"error": f"Could not retrieve stats: {e}"}

async def main():
    """Main function to demonstrate GraphRAG."""
    
    # Initialize GraphRAG system
    graphrag = GraphRAGEvaluator()
    await graphrag._check_graphrag_installation() # Ensure it's checked early
    
    # Check if index already exists
    stats = await graphrag.get_index_stats()
    if stats.get("indexing_complete", False):
        print("✅ Existing GraphRAG index found")
        print(f"📈 Index stats: {stats}")
    else:
        print("No existing index found. Creating new one...")
        
        # Prepare documents
        doc_count = await graphrag.prepare_documents()
        if doc_count == 0:
            print("❌ No documents prepared. Exiting.")
            return
        
        # Create settings
        await graphrag.create_settings_yaml()
        
        # Run indexing
        if not await graphrag.run_indexing():
            print("❌ Indexing failed. Exiting.")
            return
        
        # Get updated stats
        stats = await graphrag.get_index_stats()
        print(f"📈 Index stats: {stats}")
    
    # Test queries
    test_questions = [
        "What are the main symptoms of myocardial infarction?",
        "How is atrial fibrillation diagnosed and treated?", 
        "What are the risk factors for coronary artery disease?",
        "Explain the pathophysiology of heart failure with reduced ejection fraction.",
        "What are the contraindications for beta-blockers in cardiology?"
    ]
    
    print("\n🧪 Testing with sample questions...")
    
    # Test global search
    print("\n🌍 Testing Global Search:")
    for question in test_questions[:2]:  # Test with first 2 questions
        try:
            result = await graphrag.query_global(question)
            print(f"\n❓ Question: {question}")
            print(f"💡 Answer: {result['answer'][:300]}...")
            print(f"⏱️ Time: {result['query_time']:.2f}s")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test local search
    print("\n🎯 Testing Local Search:")
    for question in test_questions[2:4]:  # Test with next 2 questions
        try:
            result = await graphrag.query_local(question)
            print(f"\n❓ Question: {question}")
            print(f"💡 Answer: {result['answer'][:300]}...")
            print(f"⏱️ Time: {result['query_time']:.2f}s")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\nTest GraphRAG complete. Check ./graphrag_workspace for outputs.")

if __name__ == "__main__":
    # asyncio.run(main())
    # Updated to handle potential top-level await issues if script is run directly
    # For direct script execution, it's better to explicitly create and run the event loop.
    
    async def run_main_async():
        await main()

    try:
        asyncio.run(run_main_async())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("Note: main() is async. If running in an environment like Jupyter, use 'await main()' directly.")
        else:
            raise 