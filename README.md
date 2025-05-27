# Medical Reasoning Evaluation System 🏥

A minimal codebase for exploring the performance of reasoning models on medical datasets, specifically the S-MedQA dataset and similar medical question-answering tasks.

## Features

- 🤖 **OpenAI Model Evaluation**: Test GPT-3.5-turbo, GPT-4, and GPT-4-turbo on medical reasoning tasks
- 🚀 **Batch Processing**: Concurrent evaluation using OpenAI Batch API with 50% cost savings
- 📊 **Multiple Reasoning Approaches**: Direct, chain-of-thought, self-consistency, evidence-based, and differential diagnosis prompts
- 📚 **BMJ Best Practice Integration**: Download cardiology best practice PDFs for reference
- 📈 **Comprehensive Analysis**: Performance metrics by specialty, prompt type, and detailed error analysis
- 🔍 **Interactive Mode**: Easy exploration and testing
- 💾 **Automated Results Saving**: JSON, CSV, and summary reports

## Quick Start

### Prerequisites

1. Python 3.8+
2. OpenAI API key (stored in `openai_api_key.txt`)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python main.py --analyze
```

### Basic Usage

```bash
# Quick evaluation (5 samples) - synchronous
python main.py --eval --model gpt-4 --no-batch

# Batch evaluation with cost savings
python main.py --eval --batch --model gpt-4o-mini --sample-size 100

# Full evaluation on test set
python main.py --eval --full --model gpt-4 --split test

# Evaluate on filtered test set (6 key specialties)
python main.py --eval --filtered-6 --model gpt-4o-mini

# Evaluate cardiology questions only
python main.py --eval --cardiology --model gpt-3.5-turbo

# Multiple prompt types with batch processing
python main.py --eval --prompts direct chain_of_thought --model gpt-4 --batch

# Interactive mode for exploration
python main.py --interactive

# Download BMJ Best Practice cardiology PDFs
python main.py --download
```

## Dataset

The system uses the **S-MedQA** dataset, which contains:
- **Training Set**: 7,125 samples
- **Validation Set**: 899 samples  
- **Test Set**: 899 samples across 15 medical specialties
- **Test (Filtered 6)**: 459 samples from 6 key specialties

Each sample includes:
- Medical question with multiple choice options
- Correct answer
- Medical specialty annotation (15 specialties)
- Multiple annotation votes for quality assurance

### Medical Specialties Included

The dataset covers 15 medical specialties with 200+ samples each, including:
- Cardiology
- Internal Medicine
- Neurology
- Psychiatry
- Surgery
- Emergency Medicine
- And more...

### Dataset Splits

The S-MedQA dataset contains three main splits:
- **Training**: 7,125 samples across 15 medical specialties
- **Validation**: 899 samples across 15 medical specialties  
- **Test**: 899 samples across 15 medical specialties
- **Test (Filtered 6)**: 459 samples from 6 key specialties

#### Filtered Test Set

A specialized subset containing only these 6 high-priority medical specialties:
- **Cardiology**: 74 samples
- **Gastroenterology**: 86 samples
- **Infectious diseases**: 76 samples
- **Neurology**: 71 samples
- **Obstetrics and gynecology**: 89 samples
- **Pediatrics**: 63 samples

This filtered set represents 51.1% of the original test set (459/899 samples) and focuses on core medical domains for targeted evaluation.

## Processing Modes

The system supports three processing modes for different use cases:

### 1. Sequential Processing
- **Use case**: Small evaluations (<5 requests)
- **Speed**: ~1-2 requests/second
- **Cost**: Standard OpenAI pricing
- **Reliability**: Highest (simple, well-tested)

### 2. Concurrent Processing ⚡ NEW!
- **Use case**: Medium evaluations (5-100 requests)
- **Speed**: ~5-20 requests/second (5-10x faster than sequential)
- **Cost**: Same as sequential (standard OpenAI pricing)
- **Reliability**: High (with built-in rate limiting and retry logic)
- **Features**:
  - Respects OpenAI rate limits
  - Automatic retry on failures
  - Configurable concurrency levels
  - Real-time progress tracking

### 3. Batch Processing
- **Use case**: Large evaluations (100+ requests)
- **Speed**: 10 minutes to 24 hours (delayed processing)
- **Cost**: 50% cheaper than standard pricing
- **Reliability**: High (OpenAI's managed service)

## Concurrent Processing Features

### Automatic Mode Selection
The system automatically selects the optimal processing mode based on request count:
- **<5 requests**: Sequential processing
- **5-10 requests**: Concurrent processing  
- **10+ requests**: Batch processing (can be overridden)

```bash
# Auto-selection examples
python main.py --eval --sample-size 3    # → Sequential
python main.py --eval --sample-size 8    # → Concurrent  
python main.py --eval --sample-size 15   # → Batch
```

### Manual Concurrent Processing
Force concurrent processing for any evaluation:

```bash
# Basic concurrent processing
python main.py --eval --concurrent --sample-size 20

# Advanced concurrent configuration
python main.py --eval --concurrent \
  --max-concurrent 8 \
  --requests-per-minute 120 \
  --model gpt-4o-mini \
  --prompts direct chain_of_thought
```

### Interactive Concurrent Commands
Use interactive mode for easy concurrent processing:

```bash
python main.py --interactive
> concurrent              # Run interactive concurrent evaluation
> concurrent-config       # Show current concurrent settings
> processing-modes        # Compare all processing modes
```

### Configuration Options

#### Concurrent Processing Settings
```python
# config.py
CONCURRENT_CONFIG = {
    'max_concurrent_requests': 10,  # Max simultaneous requests
    'requests_per_minute': 100,     # Rate limit
    'enable_concurrent': True,      # Enable concurrent processing
    'concurrent_threshold': 5       # Min requests to trigger concurrent
}
```

#### Evaluation Settings
```python
# config.py  
EVALUATION_SETTINGS = {
    'auto_concurrent': True,        # Auto-select processing mode
    'concurrent_threshold': 5,      # Switch to concurrent at ≥5 requests
    'batch_threshold': 10          # Switch to batch at ≥10 requests
}
```

### Performance Comparison

| Mode | Speed | Cost | Best For |
|------|-------|------|----------|
| Sequential | 1-2 req/s | 1x | Small tests, debugging |
| **Concurrent** | 5-20 req/s | 1x | **Most evaluations** |
| Batch | Variable | 0.5x | Large production runs |

### Rate Limiting & Best Practices

The concurrent processor includes smart rate limiting:
- **Respects OpenAI limits**: Automatically throttles requests
- **Exponential backoff**: Handles rate limit errors gracefully
- **Conservative defaults**: Safe settings that work for all users
- **Customizable**: Adjust concurrency based on your rate limits

**Recommended settings by OpenAI tier**:
- **Free tier**: `max_concurrent=3, requests_per_minute=20`
- **Pay-as-you-go**: `max_concurrent=5, requests_per_minute=60` 
- **Usage tier 3+**: `max_concurrent=10, requests_per_minute=100`

## Evaluation Modes

### 1. Direct Evaluation
```bash
python main.py --eval --model gpt-4 --sample-size 10
```

### 2. Reasoning Comparison
```bash
python main.py --eval --prompts direct chain_of_thought self_consistency
```

### 3. Specialty-Specific Analysis
```bash
python main.py --eval --specialty "Cardiology" --full
```

### 4. Interactive Exploration
```bash
python main.py --interactive
```
Then use commands like:
- `analyze` - Dataset statistics
- `eval` - Quick 5-sample evaluation
- `cardiology` - Cardiology-specific evaluation
- `download` - Download BMJ PDFs

## Reasoning Prompt Types

### 1. Direct
Simple multiple-choice answering without explicit reasoning steps.

### 2. Chain-of-Thought
Step-by-step medical reasoning with structured thinking:

```
<think>
1. Identify key medical concepts
2. Analyze patient presentation
3. Consider differential diagnosis
4. Evaluate options
5. Select best answer
</think>

<answer>A</answer>
```

### 3. Self-Consistency
Multiple-angle analysis for robust decision making.

### 4. Evidence-Based
Focus on clinical guidelines and medical literature.

### 5. Differential Diagnosis
Systematic diagnostic reasoning approach.

## Structured Response Format

All reasoning prompts use structured tags for reliable answer extraction:

- **`<think></think>`** - Contains the model's reasoning process
- **`<answer>X</answer>`** - Contains the final answer choice (A, B, C, D, etc.)

This eliminates parsing errors and ensures 100% reliable answer extraction, compared to previous regex-based methods that were error-prone with complex medical reasoning text.

## Results and Analysis

### Automatic Result Saving
Results are automatically saved to the `results/` directory:
- `{model}_{split}_{timestamp}_detailed.json` - Complete results
- `{model}_{split}_{timestamp}_summary.json` - Performance summary
- `{model}_{split}_{timestamp}.csv` - Tabular data for analysis

### Performance Metrics
- Overall accuracy
- Performance by reasoning prompt type
- Performance by medical specialty
- Response time analysis
- Error analysis and categorization
- API usage tracking

## BMJ Best Practice Integration

Download cardiology best practice guidelines:

```bash
python main.py --download
```

This will:
1. Fetch all cardiology topics from BMJ Best Practice
2. Create a topic list for reference
3. Download available PDFs to `cardiology_pdfs/` directory

**Note**: BMJ Best Practice requires subscription access for full PDF downloads. The system will attempt to download what's publicly available.

## Configuration

### API Settings
Edit `config.py` to modify:
- Model parameters (temperature, max_tokens)
- API retry settings
- Default evaluation settings

### Model Selection
Available models:
- `gpt-4o-mini` (newest, cost-effective)
- `gpt-3.5-turbo` (default)
- `gpt-4`
- `gpt-4-turbo`

## Example Workflows

### Research Workflow
```bash
# 1. Analyze dataset
python main.py --analyze

# 2. Quick baseline evaluation
python main.py --eval --model gpt-3.5-turbo

# 3. Compare reasoning approaches
python main.py --eval --prompts direct chain_of_thought --sample-size 50

# 4. Full specialty analysis
python main.py --eval --specialty "Cardiology" --full

# 5. Compare models
python main.py --eval --model gpt-4 --full
```

### Interactive Exploration
```bash
python main.py --interactive

# Then try:
> analyze
> specialties
> eval
> cardiology
> download
```

## File Structure

```
mg-reason/
├── main.py                    # Main entry point
├── config.py                  # Configuration settings
├── data_loader.py             # Dataset loading utilities
├── model_evaluator.py         # OpenAI model evaluation
├── reasoning_prompts.py       # Prompt templates
├── bmj_pdf_downloader.py      # BMJ PDF downloader
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── .gitignore                # Git ignore file
├── S-MedQA_*.json            # Dataset files
├── results/                  # Evaluation results
└── cardiology_pdfs/          # Downloaded PDFs
```

## Performance Expectations

Based on medical QA benchmarks:
- **GPT-4o-mini**: ~60-70% accuracy (cost-effective option)
- **GPT-3.5-turbo**: ~50-60% accuracy
- **GPT-4**: ~70-80% accuracy
- **Chain-of-thought**: +5-10% improvement over direct
- **Cardiology questions**: Performance varies by complexity

## API Usage and Costs

The system tracks API usage:
- Token consumption per evaluation
- Number of API calls
- Estimated costs (varies by model)

**Cost Estimation** (approximate):
- GPT-4o-mini: ~$0.0002 per question
- GPT-3.5-turbo: ~$0.001 per question
- GPT-4: ~$0.01 per question
- Full dataset evaluation: $0.20-10 depending on model

## Contributing

This is a minimal research codebase. To extend:

1. **Add new models**: Modify `config.py` and `model_evaluator.py`
2. **New prompt types**: Add to `reasoning_prompts.py`
3. **Additional datasets**: Extend `data_loader.py`
4. **New analysis**: Enhance result processing in `model_evaluator.py`

## Limitations

- Requires OpenAI API access and credits
- BMJ PDF downloads may be limited by subscription access
- Evaluation speed limited by API rate limits
- Focus on multiple-choice questions only

## Citation

If you use this codebase in your research, please cite:

```bibtex
@software{medical_reasoning_eval,
  title={Medical Reasoning Evaluation System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/mg-reason}
}
```

## License

This project is for research and educational purposes. Please respect OpenAI's usage policies and BMJ's terms of service.

---

**Happy Medical AI Research! 🩺🤖**

## Retrieval-Augmented Generation (RAG) 🧠

The system now includes **Retrieval-Augmented Generation (RAG)** capabilities to enhance medical question answering by leveraging a knowledge base of cardiology PDFs.

Two RAG implementations are available for comparison:

1.  **LangChain RAG**: A traditional RAG approach using vector similarity search.
2.  **Microsoft GraphRAG**: An advanced RAG approach using knowledge graphs for more nuanced retrieval.

### RAG Features:

-   **Cardiology Knowledge Base**: Uses the downloaded cardiology PDFs as a source of truth.
-   **Cost-Efficient Models**: Utilizes `gpt-4o-mini` for building RAG indexes and answering questions to save costs.
-   **Persistent Indexes**: Saves and reuses generated indexes for faster subsequent runs.
-   **Side-by-Side Comparison**: Evaluate both LangChain RAG and GraphRAG on the same questions.

### RAG Commands:

#### 1. LangChain RAG

-   **Build LangChain Index**:
    ```bash
    python main.py --rag-langchain-build
    ```
    This command processes all PDFs in the `cardiology_pdfs` directory, creates text chunks, generates embeddings, and stores them in a ChromaDB vector store (`./chroma_db`).

-   **Evaluate LangChain RAG**:
    ```bash
    python main.py --rag-langchain-eval [N]
    ```
    Evaluates the LangChain RAG system on N sample cardiology questions (default: 5). It retrieves relevant context from the indexed PDFs and uses `gpt-4o-mini` to generate answers.

#### 2. Microsoft GraphRAG

-   **Build GraphRAG Index**:
    ```bash
    python main.py --rag-graphrag-build
    ```
    This command:
    1.  Converts cardiology PDFs to text files in `./graphrag_workspace/input`.
    2.  Creates a `settings.yaml` file configured for `gpt-4o-mini`.
    3.  Runs the GraphRAG indexing pipeline to create a knowledge graph and associated artifacts in `./graphrag_workspace/output`.

-   **Evaluate GraphRAG (Global Search)**:
    ```bash
    python main.py --rag-graphrag-eval-global [N]
    ```
    Evaluates GraphRAG using **global search** on N sample cardiology questions (default: 5). Global search is good for holistic questions about the entire dataset.

-   **Evaluate GraphRAG (Local Search)**:
    ```bash
    python main.py --rag-graphrag-eval-local [N]
    ```
    Evaluates GraphRAG using **local search** on N sample cardiology questions (default: 5). Local search is better for specific questions about particular entities or concepts.

### Example RAG Workflow:

1.  **Download PDFs** (if not already done):
    ```bash
    python main.py --download
    ```

2.  **Build Indexes**:
    ```bash
    python main.py --rag-langchain-build
    python main.py --rag-graphrag-build 
    ```

3.  **Run Evaluations**:
    ```bash
    python main.py --rag-langchain-eval 10
    python main.py --rag-graphrag-eval-global 10
    python main.py --rag-graphrag-eval-local 10
    ```

This will allow you to compare the performance and answer quality of traditional RAG (LangChain) versus knowledge graph-based RAG (GraphRAG) on your cardiology dataset.

## Batch Processing

