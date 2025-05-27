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

## Batch Processing 🚀

The system now supports OpenAI's Batch API for concurrent processing with significant advantages:

### Benefits
- **50% Cost Savings**: Batch requests are charged at 50% of standard rates
- **Higher Rate Limits**: Up to 250M tokens for GPT-4 batches
- **Parallel Processing**: All requests processed concurrently
- **Automatic Fallback**: Falls back to synchronous processing for small datasets

### When Batch Processing is Used
- **Automatically**: For evaluations with ≥10 total requests
- **Manual Control**: Use `--batch` or `--no-batch` flags
- **Interactive Mode**: Use `batch-eval` and `batch-full` commands
- **Demo Mode**: Use `--batch-demo` for instant testing without waiting

### Demo Mode 🎭
For testing and demonstration purposes, use demo mode to see batch processing in action instantly:

```bash
# CLI demo mode
python main.py --eval --batch-demo --sample-size 20

# Interactive demo mode
python main.py --interactive
> batch-demo
```

Demo mode simulates batch processing using synchronous calls but shows you:
- What the batch workflow looks like
- Estimated cost savings (50%)
- How results are formatted
- Processing time comparisons

### Real Batch Processing ⏰
**Important**: Real batch processing can take 10 minutes to 24 hours to complete. This is normal and expected behavior from OpenAI's Batch API. The trade-off is significant cost savings (50%) and higher rate limits.

### Batch Processing Flow
1. **Create Batch File**: Generate JSONL file with all requests
2. **Upload & Submit**: Upload to OpenAI and submit batch job
3. **Monitor Progress**: Poll status every 30 seconds
4. **Download Results**: Retrieve completed responses
5. **Process & Analyze**: Extract answers and generate metrics
6. **Cleanup**: Remove temporary files (optional)

### Configuration
Batch settings can be modified in `config.py`:
```python
BATCH_SETTINGS = {
    "enabled": True,           # Enable batch processing
    "min_batch_size": 10,      # Minimum requests for batch
    "max_batch_size": 1000,    # Maximum requests per batch  
    "poll_interval": 30,       # Status check interval (seconds)
    "max_wait_time": 86400,    # Maximum wait time (24 hours)
    "auto_cleanup": True       # Clean up batch files
}
```

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
Step-by-step medical reasoning:
1. Identify key medical concepts
2. Analyze patient presentation
3. Consider differential diagnosis
4. Evaluate options
5. Select best answer

### 3. Self-Consistency
Multiple-angle analysis for robust decision making.

### 4. Evidence-Based
Focus on clinical guidelines and medical literature.

### 5. Differential Diagnosis
Systematic diagnostic reasoning approach.

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

