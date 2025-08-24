# Ego-Id-Superego Neural Network for NLP

A PyTorch-based neural network implementation inspired by Freud's structural model of the psyche for natural language processing tasks.

## Overview

This implementation provides a novel neural network architecture that models three psychological components:

- **Id Network**: Represents instinctual, pleasure-seeking responses
- **Ego Network**: Represents rational, reality-testing processes  
- **Superego Network**: Represents moral, conscience-based evaluation

The components work together through an integration layer to provide comprehensive psychological analysis of text and human expressions.

## Features

- **Three-Component Architecture**: Separate neural networks for Id, Ego, and Superego
- **Dynamic Integration**: Weighted combination based on confidence and decision weights
- **Multiple NLP Tasks**: Sentiment analysis, emotion detection, text generation
- **Psychological Profiling**: Comprehensive analysis of psychological dynamics
- **Conflict Detection**: Identifies internal conflicts between psychological components
- **Integration with Existing Framework**: Works with HumanExpressionEvaluator

## Installation

### Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

### Optional Dependencies

For full integration with the existing NLP framework:
- `spacy` (for advanced text processing)
- Existing `HumanExpressionEvaluator` module

## Quick Start

### Basic Usage

```python
import torch
from ego_id_superego_nn import EgoIdSuperegoNeuralNetwork

# Initialize the network
network = EgoIdSuperegoNeuralNetwork(
    input_dim=384,    # Text embedding dimension
    hidden_dim=256,   # Hidden layer dimension
    output_dim=128    # Output dimension
)

# Create sample text embedding
text_embedding = torch.randn(1, 384)

# Perform psychological analysis
result = network.analyze_text_psychologically(text_embedding)

# Access results
psychological_profile = result['psychological_profile']
print(f"Dominant drive: {psychological_profile['instinctual_response']['strength']}")
print(f"Rational consistency: {psychological_profile['rational_response']['logical_consistency']}")
print(f"Moral certainty: {psychological_profile['moral_response']['ethical_certainty']}")
```

### Task-Specific Usage

```python
# Sentiment analysis
sentiment_result = network(text_embedding, task="sentiment")
sentiment_probs = sentiment_result['task_output']  # [positive, negative, neutral]

# Emotion detection  
emotion_result = network(text_embedding, task="emotion")
emotion_probs = emotion_result['task_output']  # 8 basic emotions

# Text generation
generation_result = network(text_embedding, task="generation")
generated_embedding = generation_result['task_output']
```

### Integrated Analysis

```python
from psychological_nlp_integration import PsychologicalNLPAnalyzer

# Create integrated analyzer
analyzer = PsychologicalNLPAnalyzer()

# Analyze text comprehensively
text = "I want this, but I should consider if it's right."
context = {
    'formality_level': 'informal',
    'situation': 'decision_making'
}

analysis = analyzer.analyze_text_comprehensively(text, context)

# Access integrated insights
insights = analysis['integrated_insights']
recommendations = analysis['recommendations']
```

## Architecture Details

### Id Network

The Id Network models instinctual and emotional responses:

- **Emotional Processing**: Multi-head attention for emotional focus
- **Impulse Generation**: High non-linearity with Tanh and ReLU activations
- **Desire Output**: Sigmoid-bounded outputs representing unconscious drives
- **High Reactivity**: Emphasizes immediate, pleasure-seeking responses

```python
# Key characteristics
- Input → Emotion Layer (Tanh) → Impulse Layer (ReLU) → Desire Layer (Sigmoid)
- Multi-head attention for emotional focus
- Dropout for regularization
- Confidence estimation based on desire strength
```

### Ego Network

The Ego Network represents rational, reality-testing processes:

- **Rational Processing**: Controlled ReLU activations with LayerNorm
- **Memory Integration**: Bidirectional LSTM for context processing
- **Planning Layer**: Strategic decision-making capabilities
- **Reality Testing**: Balanced Tanh outputs

```python
# Key characteristics  
- Input → Reasoning → Memory (LSTM) → Planning → Reality Testing
- LayerNorm for stable training
- Memory context integration
- Consistency-based confidence estimation
```

### Superego Network

The Superego Network models moral and ethical evaluation:

- **Moral Evaluation**: LeakyReLU for critical assessment
- **Social Norm Integration**: Social context consideration
- **Idealistic Processing**: High-level moral reasoning
- **Constraint Generation**: Moral boundaries and guidelines

```python
# Key characteristics
- Input → Moral Evaluation → Social Norms → Ideals → Constraints  
- LeakyReLU for critical evaluation
- LayerNorm for stability
- Moral certainty estimation
```

### Integration Layer

The Integration Layer combines all three components:

- **Conflict Analysis**: Detects disagreements between components
- **Dynamic Weighting**: Confidence and decision-weight based combination
- **Final Output Generation**: Integrated psychological response
- **Transparency**: Provides decision rationale and conflict levels

## Psychological Interpretation

### Component Weights

The network outputs weights indicating the influence of each component:

- **High Id Weight**: Impulsive, emotion-driven responses
- **High Ego Weight**: Rational, balanced decision-making
- **High Superego Weight**: Moral, conscience-driven responses

### Conflict Analysis

Internal conflict is measured by disagreement between components:

- **Low Conflict** (< 0.3): Harmonious psychological state
- **Medium Conflict** (0.3-0.7): Balanced internal tension  
- **High Conflict** (> 0.7): Significant psychological discord

### Psychological Profiles

The system generates comprehensive profiles including:

- **Instinctual Response**: Impulse strength, emotional intensity
- **Rational Response**: Logical consistency, planning depth
- **Moral Response**: Ethical certainty, idealistic level
- **Psychological Dynamics**: Conflict level, decision clarity, component harmony

## Applications

### Text Analysis

- **Sentiment Analysis**: Enhanced with psychological depth
- **Emotion Detection**: Multi-faceted emotional understanding
- **Intent Recognition**: Psychological motivation analysis
- **Personality Assessment**: Text-based psychological profiling

### Content Generation

- **Psychologically-Informed Text Generation**: Content reflecting specific psychological states
- **Dialogue Systems**: More nuanced conversational AI
- **Creative Writing**: Psychologically complex character development

### Clinical Applications

- **Therapeutic Text Analysis**: Understanding psychological patterns in text
- **Mental Health Screening**: Early detection through language patterns
- **Treatment Progress**: Monitoring psychological changes over time

## Testing and Validation

Run the comprehensive test suite:

```bash
python test_ego_id_superego_nn.py
```

The test suite includes:

- **Component Tests**: Individual network functionality
- **Integration Tests**: Multi-component coordination
- **Behavioral Tests**: Consistency and variation analysis
- **Performance Tests**: Speed and memory usage

### Test Results

All tests pass, validating:

- ✅ Component initialization and basic functionality
- ✅ Integration layer coordination
- ✅ Task-specific outputs
- ✅ Psychological analysis generation
- ✅ Network consistency and determinism
- ✅ Component influence variation

## Performance Considerations

### Computational Complexity

- **Id Network**: O(n²) for attention, O(n) for feedforward
- **Ego Network**: O(n²) for LSTM, O(n) for feedforward  
- **Superego Network**: O(n) for feedforward layers
- **Integration**: O(n) for combination and analysis

### Memory Usage

- **Base Model**: ~2MB for default parameters
- **Batch Processing**: Linear scaling with batch size
- **Memory-Efficient**: Supports large-scale text analysis

### Optimization Tips

1. **Batch Size**: Use larger batches for better GPU utilization
2. **Mixed Precision**: Enable for faster training on modern GPUs
3. **Gradient Accumulation**: For training with limited memory
4. **Caching**: Cache text embeddings for repeated analysis

## Advanced Usage

### Custom Training

```python
import torch.optim as optim
import torch.nn as nn

# Initialize network
network = EgoIdSuperegoNeuralNetwork(input_dim=384)

# Setup training
optimizer = optim.Adam(network.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        result = network(batch['text_embeddings'], task="sentiment")
        loss = criterion(result['task_output'], batch['labels'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

### Custom Components

Extend the architecture with custom psychological components:

```python
class CustomPsycheComponent(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.custom_layer = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return PsycheOutput(
            component=PsycheComponent.CUSTOM,
            hidden_state=self.custom_layer(x),
            confidence=0.5,
            activation_pattern=x,
            decision_weight=0.5
        )
```

## Comparison with Existing Approaches

### Traditional NLP Models

| Feature | Traditional | Ego-Id-Superego |
|---------|-------------|-----------------|
| Psychological Depth | Limited | Comprehensive |
| Interpretability | Low | High |
| Decision Rationale | None | Detailed |
| Conflict Detection | No | Yes |
| Multi-dimensional Analysis | No | Yes |

### Advantages

1. **Psychological Interpretability**: Clear mapping to psychological theory
2. **Conflict Detection**: Identifies internal inconsistencies
3. **Multi-perspective Analysis**: Three distinct viewpoints on text
4. **Decision Transparency**: Explains why certain outputs were generated
5. **Adaptive Integration**: Dynamic weighting based on confidence

### Limitations

1. **Computational Overhead**: More complex than single-network approaches
2. **Training Complexity**: Requires coordination between multiple components
3. **Theoretical Basis**: Dependent on Freudian psychological theory
4. **Validation Challenges**: Difficult to validate psychological accuracy

## Future Development

### Planned Enhancements

1. **Multi-Language Support**: Extend to non-English texts
2. **Real-time Processing**: Optimize for streaming text analysis
3. **Personalization**: Adapt to individual psychological profiles
4. **Clinical Validation**: Validate against psychological assessments

### Research Directions

1. **Empirical Validation**: Compare with human psychological assessments
2. **Cross-Cultural Analysis**: Adapt to different cultural contexts
3. **Temporal Dynamics**: Model psychological state changes over time
4. **Integration with Other Theories**: Incorporate additional psychological models

## Contributing

Contributions are welcome! Areas of interest:

- **Performance Optimization**: Speed and memory improvements
- **New Applications**: Novel use cases and implementations
- **Validation Studies**: Empirical validation of psychological accuracy
- **Documentation**: Improved examples and tutorials

## License

This implementation is provided under the MIT License for educational and research purposes.

## References

1. Freud, S. (1923). The Ego and the Id
2. Modern Neural Network Architectures
3. Attention Mechanisms in Deep Learning
4. Psychological Assessment in NLP

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in the repository.

---

*This implementation bridges the gap between computational psychology and natural language processing, providing a novel framework for psychologically-informed text analysis.*