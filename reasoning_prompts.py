"""Reasoning prompt templates for medical question answering."""
from typing import Dict, List, Optional

class ReasoningPrompts:
    """Collection of reasoning prompt templates for medical QA."""
    
    @staticmethod
    def direct_prompt(question: str, options: List[str]) -> str:
        """Direct answering prompt without explicit reasoning steps."""
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"""You are a medical expert. Answer the following multiple-choice question by selecting the best option.

Question: {question}

Options:
{formatted_options}

Please provide your answer in the following format:
<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""
    
    @staticmethod
    def chain_of_thought_prompt(question: str, options: List[str]) -> str:
        """Chain-of-thought reasoning prompt."""
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"""You are a medical expert. Answer the following multiple-choice question using step-by-step reasoning.

Question: {question}

Options:
{formatted_options}

Please structure your response as follows:

<think>
Think through this step-by-step:
1. First, identify the key medical concepts and clinical scenario
2. Analyze the patient's presentation, symptoms, and any relevant findings
3. Consider the differential diagnosis and pathophysiology
4. Evaluate each option based on medical knowledge and guidelines
5. Select the most appropriate answer
</think>

<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""
    
    @staticmethod
    def self_consistency_prompt(question: str, options: List[str]) -> str:
        """Self-consistency reasoning prompt for multiple attempts."""
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"""You are a medical expert. Answer the following multiple-choice question by thinking through it carefully.

Question: {question}

Options:
{formatted_options}

Please structure your response as follows:

<think>
Consider this question from multiple angles:
- What is the most likely diagnosis or appropriate treatment?
- What are the key clinical features that point to the answer?
- Which option best aligns with current medical guidelines and evidence?
</think>

<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""
    
    @staticmethod
    def evidence_based_prompt(question: str, options: List[str]) -> str:
        """Evidence-based reasoning prompt focusing on clinical guidelines."""
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"""You are a medical expert familiar with current clinical guidelines and evidence-based medicine. Answer the following multiple-choice question.

Question: {question}

Options:
{formatted_options}

Please structure your response as follows:

<think>
Base your answer on:
1. Current clinical guidelines and recommendations
2. Evidence from medical literature and studies
3. Standard of care practices
4. Risk-benefit analysis
5. Patient safety considerations
</think>

<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""
    
    @staticmethod
    def differential_diagnosis_prompt(question: str, options: List[str]) -> str:
        """Differential diagnosis focused prompt."""
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"""You are a medical expert specializing in diagnostic reasoning. Answer the following multiple-choice question.

Question: {question}

Options:
{formatted_options}

Please structure your response as follows:

<think>
Approach this systematically:
1. Identify the chief complaint and key clinical features
2. Generate a differential diagnosis list
3. Consider which findings support or refute each possibility
4. Apply clinical reasoning to narrow down to the most likely diagnosis/treatment
5. Select the option that best fits the clinical picture
</think>

<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""

    @staticmethod
    def direct_prompt_rag(question: str, options: List[str], context: str) -> str:
        """Direct answering prompt augmented with RAG context."""
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"""You are a medical expert. Answer the following multiple-choice question by selecting the best option, using the provided context from medical literature.

Context:
{context}

Question: {question}

Options:
{formatted_options}

Please provide your answer in the following format:
<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""
    
    @staticmethod
    def chain_of_thought_prompt_rag(question: str, options: List[str], context: str) -> str:
        """Chain-of-thought reasoning prompt augmented with RAG context."""
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"""You are a medical expert. Answer the following multiple-choice question using step-by-step reasoning, informed by the provided context from medical literature.

Context:
{context}

Question: {question}

Options:
{formatted_options}

Please structure your response as follows:

<think>
Think through this step-by-step, incorporating information from the context:
1. First, identify the key medical concepts and clinical scenario from the question and context.
2. Analyze the patient's presentation, symptoms, and any relevant findings, cross-referencing with the context.
3. Consider the differential diagnosis and pathophysiology, using the context to support your reasoning.
4. Evaluate each option based on medical knowledge, guidelines, and the provided context.
5. Select the most appropriate answer.
</think>

<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""
    
    @staticmethod
    def self_consistency_prompt_rag(question: str, options: List[str], context: str) -> str:
        """Self-consistency reasoning prompt augmented with RAG context."""
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"""You are a medical expert. Answer the following multiple-choice question by thinking through it carefully, using the provided context.

Context:
{context}

Question: {question}

Options:
{formatted_options}

Please structure your response as follows:

<think>
Consider this question from multiple angles, informed by the context:
- What is the most likely diagnosis or appropriate treatment, based on the context?
- What are the key clinical features that point to the answer, supported by the context?
- Which option best aligns with current medical guidelines and evidence as presented in the context?
</think>

<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""
    
    @staticmethod
    def evidence_based_prompt_rag(question: str, options: List[str], context: str) -> str:
        """Evidence-based reasoning prompt augmented with RAG context."""
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"""You are a medical expert familiar with current clinical guidelines and evidence-based medicine. Answer the following multiple-choice question using the provided context.

Context:
{context}

Question: {question}

Options:
{formatted_options}

Please structure your response as follows:

<think>
Base your answer on the provided context and your expertise:
1. Current clinical guidelines and recommendations found in or supported by the context.
2. Evidence from medical literature and studies as presented in the context.
3. Standard of care practices, referenced against the context.
4. Risk-benefit analysis informed by the context.
5. Patient safety considerations highlighted in the context.
</think>

<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""
    
    @staticmethod
    def differential_diagnosis_prompt_rag(question: str, options: List[str], context: str) -> str:
        """Differential diagnosis focused prompt augmented with RAG context."""
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"""You are a medical expert specializing in diagnostic reasoning. Answer the following multiple-choice question using the provided context.

Context:
{context}

Question: {question}

Options:
{formatted_options}

Please structure your response as follows:

<think>
Approach this systematically, using the context:
1. Identify the chief complaint and key clinical features from the question and context.
2. Generate a differential diagnosis list, supported by information in the context.
3. Consider which findings in the context support or refute each possibility.
4. Apply clinical reasoning to narrow down to the most likely diagnosis/treatment, based on the context.
5. Select the option that best fits the clinical picture and the provided context.
</think>

<answer>X</answer>

Where X is the letter (A, B, C, D, etc.) of the correct option."""

class PromptTemplates:
    """Factory class for getting different prompt types."""
    
    PROMPT_TYPES = {
        "direct": ReasoningPrompts.direct_prompt,
        "chain_of_thought": ReasoningPrompts.chain_of_thought_prompt,
        "self_consistency": ReasoningPrompts.self_consistency_prompt,
        "evidence_based": ReasoningPrompts.evidence_based_prompt,
        "differential_diagnosis": ReasoningPrompts.differential_diagnosis_prompt,
        "direct_rag": ReasoningPrompts.direct_prompt_rag,
        "chain_of_thought_rag": ReasoningPrompts.chain_of_thought_prompt_rag,
        "self_consistency_rag": ReasoningPrompts.self_consistency_prompt_rag,
        "evidence_based_rag": ReasoningPrompts.evidence_based_prompt_rag,
        "differential_diagnosis_rag": ReasoningPrompts.differential_diagnosis_prompt_rag,
    }
    
    @classmethod
    def get_prompt(cls, prompt_type: str, question: str, options: List[str], context: Optional[str] = None) -> str:
        """Get a formatted prompt of the specified type, with optional RAG context."""
        if prompt_type not in cls.PROMPT_TYPES:
            raise ValueError(f"Unknown prompt type: {prompt_type}. Available types: {list(cls.PROMPT_TYPES.keys())}")
        
        prompt_func = cls.PROMPT_TYPES[prompt_type]
        
        if "_rag" in prompt_type:
            if context is None:
                raise ValueError(f"Context is required for RAG prompt type: {prompt_type}")
            return prompt_func(question, options, context)
        else:
            return prompt_func(question, options)
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available prompt types."""
        return list(cls.PROMPT_TYPES.keys())

# Example few-shot prompts for improved performance
class FewShotExamples:
    """Few-shot examples for medical reasoning."""
    
    CARDIOLOGY_EXAMPLES = [
        {
            "question": "A 65-year-old man presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, and aVF. What is the most likely diagnosis?",
            "options": ["Anterior STEMI", "Inferior STEMI", "Unstable angina", "Pulmonary embolism"],
            "reasoning": "The ST elevation in leads II, III, and aVF indicates inferior wall involvement, pointing to inferior STEMI.",
            "answer": "B"
        },
        {
            "question": "A patient with heart failure is on ACE inhibitors but still symptomatic. What medication should be added next?",
            "options": ["Calcium channel blocker", "Beta-blocker", "Digoxin", "Thiazide diuretic"],
            "reasoning": "According to heart failure guidelines, beta-blockers are the next step after ACE inhibitors in symptomatic heart failure.",
            "answer": "B"
        }
    ]
    
    @classmethod
    def get_few_shot_prompt(cls, examples: List[Dict], question: str, options: List[str]) -> str:
        """Create a few-shot prompt with examples."""
        prompt = "You are a medical expert. Here are some examples of medical reasoning:\n\n"
        
        for i, example in enumerate(examples):
            formatted_options = ""
            for j, option in enumerate(example["options"]):
                formatted_options += f"{chr(65 + j)}. {option}\n"
            
            prompt += f"Example {i+1}:\n"
            prompt += f"Question: {example['question']}\n"
            prompt += f"Options:\n{formatted_options}"
            prompt += f"Reasoning: {example['reasoning']}\n"
            prompt += f"Answer: {example['answer']}\n\n"
        
        # Add the actual question
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        prompt += f"Now answer this question:\n"
        prompt += f"Question: {question}\n"
        prompt += f"Options:\n{formatted_options}"
        prompt += f"Reasoning:"
        
        return prompt

# Example usage
if __name__ == "__main__":
    # Test different prompt types
    sample_question = "A 45-year-old patient presents with chest pain. What is the most appropriate initial test?"
    sample_options = ["Chest X-ray", "ECG", "Echocardiogram", "Stress test"]
    sample_context = "Recent guidelines suggest ECG as the first-line investigation for acute chest pain to rule out myocardial infarction. A chest X-ray might be considered if pulmonary causes are suspected. Echocardiogram is useful for structural assessment but not typically initial for undifferentiated chest pain. Stress tests are for stable angina."
    
    for prompt_type in PromptTemplates.get_available_types():
        print(f"\n{'='*50}")
        print(f"PROMPT TYPE: {prompt_type.upper()}")
        print(f"{'='*50}")
        try:
            if "_rag" in prompt_type:
                prompt = PromptTemplates.get_prompt(prompt_type, sample_question, sample_options, context=sample_context)
            else:
                prompt = PromptTemplates.get_prompt(prompt_type, sample_question, sample_options)
            print(prompt)
        except ValueError as e:
            print(f"Note: Skipping RAG prompt in standalone test as context is specific - {e}")
        print() 