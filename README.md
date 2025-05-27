# Medical Reasoning Evaluation System 🏥

A minimal codebase for exploring the performance of reasoning models on medical datasets, specifically the S-MedQA dataset and similar medical question-answering tasks.

## Features

- 🤖 **OpenAI Model Evaluation**: Test GPT-3.5-turbo, GPT-4, and GPT-4-turbo on medical reasoning tasks
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
# Quick evaluation (5 samples)
python main.py --eval

# Full evaluation with multiple prompt types
python main.py --eval --full --prompts direct chain_of_thought

# Evaluate cardiology questions specifically
python main.py --eval --cardiology

# Interactive mode for exploration
python main.py --interactive

# Download BMJ Best Practice cardiology PDFs
python main.py --download
```

## Dataset

The system uses the **S-MedQA** dataset, which contains:
- **Training Set**: 7,125 samples
- **Validation Set**: 899 samples  
- **Test Set**: 893 samples

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
- GPT-3.5-turbo: ~$0.001 per question
- GPT-4: ~$0.01 per question
- Full dataset evaluation: $1-10 depending on model

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

