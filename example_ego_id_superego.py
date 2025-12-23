#!/usr/bin/env python3
"""
Simple example demonstrating the Ego-Id-Superego Neural Network

This script shows how to use the psychological neural network for basic text analysis.
Run this file to see the network in action!
"""

import torch
import numpy as np
from ego_id_superego_nn import EgoIdSuperegoNeuralNetwork


def simple_text_to_embedding(text: str, embedding_dim: int = 384) -> torch.Tensor:
    """
    Convert text to a simple embedding for demonstration.
    In a real application, you'd use BERT, GPT, or similar embeddings.
    """
    # Create a deterministic embedding based on text hash
    text_hash = hash(text.lower()) % (2**31)
    np.random.seed(text_hash)
    
    # Generate normalized embedding
    embedding = np.random.randn(embedding_dim)
    embedding = embedding / np.linalg.norm(embedding)
    
    return torch.from_numpy(embedding).float().unsqueeze(0)


def analyze_example_texts():
    """Analyze a set of example texts with different psychological characteristics"""
    
    print("🧠 Ego-Id-Superego Neural Network Example")
    print("=" * 50)
    
    # Initialize the network
    network = EgoIdSuperegoNeuralNetwork(input_dim=384, hidden_dim=256, output_dim=128)
    network.eval()  # Set to evaluation mode
    
    # Example texts with different psychological profiles
    example_texts = [
        {
            'text': "I NEED this chocolate cake right now! I don't care about calories!",
            'expected_profile': 'High Id (impulsive, pleasure-seeking)'
        },
        {
            'text': "Let me research the pros and cons before making this important decision.",
            'expected_profile': 'High Ego (rational, balanced)'
        },
        {
            'text': "We must always do what's morally right, even if it's difficult.",
            'expected_profile': 'High Superego (moral, principled)'
        },
        {
            'text': "I want it, but I should consider if it's the right thing to do.",
            'expected_profile': 'Balanced (all components active)'
        }
    ]
    
    print(f"Analyzing {len(example_texts)} example texts...\n")
    
    # Analyze each text
    for i, example in enumerate(example_texts, 1):
        text = example['text']
        expected = example['expected_profile']
        
        print(f"Example {i}: {expected}")
        print(f"Text: \"{text}\"")
        print("-" * 40)
        
        # Convert text to embedding
        text_embedding = simple_text_to_embedding(text)
        
        # Perform psychological analysis
        with torch.no_grad():
            analysis = network.analyze_text_psychologically(text_embedding)
        
        # Extract key information
        integrated = analysis['integrated_response']
        rationale = integrated.decision_rationale
        profile = analysis['psychological_profile']
        
        # Show component weights
        print("Component Influence:")
        print(f"  🎭 Id (Instinctual):  {rationale['id_weight']:.3f}")
        print(f"  🧠 Ego (Rational):    {rationale['ego_weight']:.3f}")
        print(f"  👼 Superego (Moral):  {rationale['superego_weight']:.3f}")
        
        # Show psychological insights
        print("\nPsychological Profile:")
        instinctual = profile['instinctual_response']
        rational = profile['rational_response']
        moral = profile['moral_response']
        dynamics = profile['psychological_dynamics']
        
        print(f"  Emotional Intensity:   {instinctual['emotional_intensity']:.3f}")
        print(f"  Rational Consistency:  {rational['logical_consistency']:.3f}")
        print(f"  Moral Certainty:       {moral['ethical_certainty']:.3f}")
        print(f"  Internal Conflict:     {dynamics['internal_conflict']:.3f}")
        print(f"  Decision Clarity:      {dynamics['decision_clarity']:.3f}")
        
        # Determine dominant component
        weights = {
            'Id': rationale['id_weight'],
            'Ego': rationale['ego_weight'],
            'Superego': rationale['superego_weight']
        }
        dominant = max(weights.items(), key=lambda x: x[1])
        
        print(f"\n🎯 Dominant Component: {dominant[0]} ({dominant[1]:.3f})")
        
        # Show conflict level interpretation
        conflict = dynamics['internal_conflict']
        if conflict < 0.3:
            conflict_desc = "Low (Harmonious)"
        elif conflict < 0.7:
            conflict_desc = "Medium (Balanced tension)"
        else:
            conflict_desc = "High (Significant discord)"
        
        print(f"⚡ Conflict Level: {conflict_desc}")
        print("\n" + "="*50 + "\n")


def demonstrate_tasks():
    """Demonstrate different NLP tasks"""
    
    print("🔧 Task-Specific Demonstrations")
    print("=" * 40)
    
    network = EgoIdSuperegoNeuralNetwork(input_dim=384)
    network.eval()
    
    sample_text = "I'm feeling really excited about this new opportunity!"
    text_embedding = simple_text_to_embedding(sample_text)
    
    print(f"Sample Text: \"{sample_text}\"")
    print()
    
    # Demonstrate different tasks
    tasks = [
        ('sentiment', 'Sentiment Analysis', 3, ['Positive', 'Negative', 'Neutral']),
        ('emotion', 'Emotion Detection', 8, ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust', 'Trust', 'Anticipation']),
    ]
    
    with torch.no_grad():
        for task_name, task_desc, output_size, labels in tasks:
            print(f"📊 {task_desc}:")
            
            result = network(text_embedding, task=task_name)
            output = result['task_output']
            
            # Show top predictions
            probs = output[0].numpy()
            top_indices = np.argsort(probs)[::-1][:3]  # Top 3
            
            for i, idx in enumerate(top_indices):
                if idx < len(labels):
                    label = labels[idx]
                    prob = probs[idx]
                    print(f"  {i+1}. {label}: {prob:.3f}")
            print()
    
    # Demonstrate text generation
    print("✍️  Text Generation:")
    with torch.no_grad():
        result = network(text_embedding, task="generation")
        generation_output = result['task_output']
        print(f"  Generated embedding shape: {generation_output.shape}")
        print(f"  Sample values: {generation_output[0][:5].tolist()}")
    
    print("\n" + "="*40)


def main():
    """Main example function"""
    
    # Set seeds for reproducible results
    torch.manual_seed(42)
    np.random.seed(42)
    
    print()
    print("🚀 Welcome to the Ego-Id-Superego Neural Network!")
    print("This example demonstrates psychological analysis of text.")
    print()
    
    # Run the main analysis
    analyze_example_texts()
    
    # Demonstrate different tasks
    demonstrate_tasks()
    
    print("✅ Example completed!")
    print()
    print("💡 Try modifying the example texts to see how the psychological")
    print("   analysis changes. You can also experiment with different")
    print("   network parameters and tasks.")
    print()
    print("📖 For more information, see EGO_ID_SUPEREGO_README.md")


if __name__ == "__main__":
    main()