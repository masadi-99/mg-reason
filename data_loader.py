"""Data loading utilities for medical reasoning datasets."""
import json
import pandas as pd
import random
from typing import Dict, List, Optional, Tuple
from config import DATASET_PATHS, EVALUATION_SETTINGS

class MedQADataLoader:
    """Data loader for S-MedQA dataset."""
    
    def __init__(self):
        self.data = {}
        self.load_all_splits()
    
    def load_all_splits(self) -> None:
        """Load all dataset splits (train, validation, test)."""
        for split, path in DATASET_PATHS.items():
            self.data[split] = self.load_split(path)
            print(f"Loaded {len(self.data[split])} samples from {split} set")
    
    def load_split(self, file_path: str) -> List[Dict]:
        """Load a single dataset split."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file {file_path} not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {file_path}")
    
    def get_split(self, split: str, sample_size: Optional[int] = None, 
                  specialty_filter: Optional[str] = None, 
                  random_seed: int = 42) -> List[Dict]:
        """
        Get a specific dataset split with optional sampling and filtering.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            sample_size: Number of samples to return (None for all)
            specialty_filter: Filter by medical specialty
            random_seed: Random seed for sampling
        
        Returns:
            List of samples
        """
        if split not in self.data:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(self.data.keys())}")
        
        data = self.data[split].copy()
        
        # Filter by specialty if specified
        if specialty_filter:
            data = [sample for sample in data 
                   if self.get_sample_specialty(sample).lower() == specialty_filter.lower()]
        
        # Sample if specified
        if sample_size and sample_size < len(data):
            random.seed(random_seed)
            data = random.sample(data, sample_size)
        
        return data
    
    def get_specialties(self, split: str = 'train') -> Dict[str, int]:
        """Get distribution of medical specialties in a split."""
        specialties = {}
        for sample in self.data[split]:
            # Handle different field names for specialty across splits
            specialty = sample.get('Voting_3') or sample.get('Specialty', 'Unknown')
            specialties[specialty] = specialties.get(specialty, 0) + 1
        return dict(sorted(specialties.items(), key=lambda x: x[1], reverse=True))
    
    def format_question(self, sample: Dict) -> str:
        """Format a question sample for model input."""
        question = sample['Question']
        options = sample['Options']
        
        formatted_options = ""
        for i, option in enumerate(options):
            formatted_options += f"{chr(65 + i)}. {option}\n"
        
        return f"Question: {question}\n\nOptions:\n{formatted_options}\nAnswer:"
    
    def get_correct_answer(self, sample: Dict) -> str:
        """Get the correct answer for a sample."""
        return sample['Answer']
    
    def get_answer_choice(self, sample: Dict) -> str:
        """Get the answer choice letter (A, B, C, D, etc.)."""
        correct_answer = sample['Answer']
        options = sample['Options']
        
        try:
            index = options.index(correct_answer)
            return chr(65 + index)  # Convert to A, B, C, D...
        except ValueError:
            return "Unknown"
    
    def get_sample_specialty(self, sample: Dict) -> str:
        """Get the specialty for a sample, handling different field names."""
        return sample.get('Voting_3') or sample.get('Specialty', 'Unknown')
    
    def get_dataset_stats(self) -> Dict:
        """Get comprehensive dataset statistics."""
        stats = {}
        
        for split in self.data:
            split_data = self.data[split]
            stats[split] = {
                'total_samples': len(split_data),
                'specialties': self.get_specialties(split),
                'avg_question_length': sum(len(s['Question']) for s in split_data) / len(split_data),
                'avg_options_count': sum(len(s['Options']) for s in split_data) / len(split_data)
            }
        
        return stats
    
    def export_to_csv(self, split: str, output_path: str) -> None:
        """Export a dataset split to CSV format."""
        data = self.get_split(split)
        
        # Flatten the data for CSV export
        flattened_data = []
        for sample in data:
            row = {
                'question': sample['Question'],
                'specialty': self.get_sample_specialty(sample),
                'answer': sample['Answer'],
                'answer_choice': self.get_answer_choice(sample)
            }
            # Add options as separate columns
            for i, option in enumerate(sample['Options']):
                row[f'option_{chr(65 + i)}'] = option
            
            flattened_data.append(row)
        
        df = pd.DataFrame(flattened_data)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} samples to {output_path}")

# Example usage
if __name__ == "__main__":
    loader = MedQADataLoader()
    
    # Print dataset statistics
    stats = loader.get_dataset_stats()
    for split, stat in stats.items():
        print(f"\n{split.upper()} SPLIT:")
        print(f"  Total samples: {stat['total_samples']}")
        print(f"  Average question length: {stat['avg_question_length']:.1f} characters")
        print(f"  Top 5 specialties:")
        for specialty, count in list(stat['specialties'].items())[:5]:
            print(f"    {specialty}: {count}")
    
    # Example: Get cardiology questions from test set
    cardiology_questions = loader.get_split('test', specialty_filter='Cardiology')
    print(f"\nFound {len(cardiology_questions)} cardiology questions in test set")
    
    if cardiology_questions:
        print("\nExample cardiology question:")
        example = cardiology_questions[0]
        print(loader.format_question(example))
        print(f"Correct answer: {loader.get_answer_choice(example)}) {loader.get_correct_answer(example)}") 