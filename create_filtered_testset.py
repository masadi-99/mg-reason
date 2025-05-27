#!/usr/bin/env python3
"""
Create a filtered test set with only specific medical specialties.
"""
import json
from data_loader import MedQADataLoader

def create_filtered_testset():
    """Create a filtered test set with only specified specialties."""
    
    # Target specialties (using exact names from the dataset)
    target_specialties = {
        'Cardiology',
        'Gastroenterology', 
        'Infectious diseases',
        'Neurology',
        'Obstetrics and gynecology',
        'Pediatrics'
    }
    
    print("🔬 Creating Filtered Test Set")
    print("=" * 50)
    print(f"Target specialties: {', '.join(target_specialties)}")
    
    # Load the data
    loader = MedQADataLoader()
    test_data = loader.get_split('test')
    
    # Filter for target specialties
    filtered_samples = []
    specialty_counts = {}
    
    for sample in test_data:
        specialty = loader.get_sample_specialty(sample)
        
        if specialty in target_specialties:
            filtered_samples.append(sample)
            specialty_counts[specialty] = specialty_counts.get(specialty, 0) + 1
    
    # Sort by specialty for organized output
    filtered_samples.sort(key=lambda x: loader.get_sample_specialty(x))
    
    # Save the filtered test set
    output_file = "S-MedQA_test_filtered_6specialties.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_samples, f, indent=2, ensure_ascii=False)
    
    # Generate statistics
    print(f"\n📊 Filtered Test Set Statistics")
    print("=" * 50)
    print(f"Original test set size: {len(test_data)}")
    print(f"Filtered test set size: {len(filtered_samples)}")
    print(f"Reduction: {len(test_data) - len(filtered_samples)} samples removed")
    print(f"Retention rate: {len(filtered_samples)/len(test_data)*100:.1f}%")
    
    print(f"\n🏥 Specialty Distribution in Filtered Set:")
    print("-" * 50)
    total_filtered = 0
    for specialty in sorted(specialty_counts.keys()):
        count = specialty_counts[specialty]
        total_filtered += count
        print(f"{specialty:30}: {count:3d} samples")
    
    print(f"\nTotal filtered samples: {total_filtered}")
    print(f"Saved to: {output_file}")
    
    # Create a summary file
    summary = {
        "description": "Filtered S-MedQA test set with 6 key medical specialties",
        "specialties": list(target_specialties),
        "original_size": len(test_data),
        "filtered_size": len(filtered_samples),
        "retention_rate": len(filtered_samples)/len(test_data),
        "specialty_counts": specialty_counts,
        "created_from": "S-MedQA_test.json"
    }
    
    summary_file = "S-MedQA_test_filtered_6specialties_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    
    # Show some sample questions
    print(f"\n📝 Sample Questions from Each Specialty:")
    print("-" * 50)
    shown_specialties = set()
    for sample in filtered_samples:
        specialty = loader.get_sample_specialty(sample)
        if specialty not in shown_specialties:
            question = sample['Question'][:150] + "..." if len(sample['Question']) > 150 else sample['Question']
            print(f"\n{specialty}:")
            print(f"  Q: {question}")
            print(f"  Options: {', '.join(sample['Options'])}")
            print(f"  Answer: {sample['Answer']}")
            shown_specialties.add(specialty)
            
            if len(shown_specialties) >= 3:  # Show 3 examples
                break
    
    return output_file, len(filtered_samples), specialty_counts

if __name__ == "__main__":
    create_filtered_testset() 