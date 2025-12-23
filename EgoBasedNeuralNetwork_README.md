# Ego-Based Neural Network (自我導向神經網路)

## Overview / 概述

This repository implements an ego-based neural network that incorporates psychological concepts into machine learning. The network exhibits "ego" behavior through three key mechanisms:

本代碼庫實現了一個將心理學概念融入機器學習的自我導向神經網路。該網路通過三個關鍵機制展現「自我」行為：

1. **Confirmation Bias (確認偏誤)**: The network gives higher weight to examples that agree with its previous predictions
2. **Parameter Inertia (參數慣性)**: The network resists changing too far from its initial parameters
3. **Output Inertia (輸出慣性)**: The network tries to maintain consistency with its previous outputs

## Key Concepts / 關鍵概念

### 1. Confirmation Bias (確認偏誤)

The network implements confirmation bias by adjusting the loss weights based on whether previous predictions agreed with the ground truth:

```python
# Higher weight for examples where we previously agreed with truth
# 對之前預測正確的例子給予更高權重
if prev_predictions == ground_truth:
    weight = α  # α = 0.7 (higher weight)
else:
    weight = 1 - α  # 1 - α = 0.3 (lower weight)
```

This mimics human tendency to pay more attention to information that confirms existing beliefs.

### 2. Parameter Inertia (參數慣性)

The network includes a regularization term that penalizes parameters for deviating too far from their initial values:

```python
param_loss = λ_p * Σ(θ_current - θ_initial)²
```

Where `λ_p = 5e-3` controls the strength of parameter inertia.

### 3. Output Inertia (輸出慣性)

The network includes a term that encourages consistency with previous outputs:

```python
output_loss = λ_o * MSE(logits_current, logits_previous)
```

Where `λ_o = 1e-2` controls the strength of output inertia.

## Implementation Files / 實現文件

### 1. `EgoBasedNeuralNetwork.py`
- **Object-oriented implementation** with separate classes for the model and trainer
- **面向對象的實現**，模型和訓練器使用獨立的類
- More modular and extensible design
- Includes comprehensive documentation and type hints

### 2. `OriginalEgoNetwork.py`
- **Direct translation** of the original issue code
- **原始問題代碼的直接翻譯**
- Maintains the exact same structure and logic
- Closer to the mathematical formulation

## Usage Examples / 使用示例

### Using the Object-Oriented Implementation

```python
from Python.EgoBasedNeuralNetwork import EgoMLP, EgoBasedTrainer, create_toy_dataset

# Create dataset
x, y = create_toy_dataset(n_samples=512, input_dim=20, n_classes=3)

# Create model
model = EgoMLP(input_dim=20, output_dim=3)

# Create trainer with ego parameters
trainer = EgoBasedTrainer(
    model=model,
    lambda_param=5e-3,   # Parameter inertia
    lambda_output=1e-2,  # Output inertia
    alpha=0.7,           # Confirmation bias
    learning_rate=1e-3
)

# Train the model
loss_history = trainer.train(x, y, epochs=200)
```

### Using the Original Implementation

```python
from Python.OriginalEgoNetwork import train_ego_based_neural_network

# Run the complete training pipeline
model, history = train_ego_based_neural_network()
```

## Mathematical Formulation / 數學公式

The total loss function combines three components:

總損失函數結合三個組成部分：

```
L_total = L_task + L_param + L_output

Where:
L_task = Σ(w_i * CrossEntropy(ŷ_i, y_i))  # Weighted task loss
L_param = λ_p * Σ(θ - θ_0)²              # Parameter inertia
L_output = λ_o * MSE(ŷ_t, ŷ_{t-1})       # Output inertia

And the confirmation bias weights are:
w_i = α if prev_pred_i == y_i else (1-α)
```

## Hyperparameters / 超參數

| Parameter | Value | Description |
|-----------|-------|-------------|
| `λ_p` | 5e-3 | Parameter inertia strength (參數慣性強度) |
| `λ_o` | 1e-2 | Output inertia strength (輸出慣性強度) |
| `α` | 0.7 | Confirmation bias weight (確認偏誤權重) |
| Learning Rate | 1e-3 | Adam optimizer learning rate |

## Results / 結果

The ego-based neural network demonstrates:
- **Stable learning** with controlled parameter drift
- **Confirmation bias effects** in loss weighting
- **Output consistency** across training epochs
- **Final accuracy** around 68-71% on toy dataset

自我導向神經網路展現了：
- **穩定學習**，參數漂移受控
- **確認偏誤效應**在損失加權中體現
- **輸出一致性**貫穿訓練階段
- **最終準確率**在玩具數據集上約為68-71%

## Philosophical Implications / 哲學含義

This implementation explores the intersection of psychology and machine learning by:

1. **Modeling human-like biases** in artificial systems
2. **Questioning objective learning** vs. ego-driven learning
3. **Exploring the balance** between adaptation and consistency
4. **Investigating confirmation bias** as both limitation and feature

該實現通過以下方式探索心理學與機器學習的交集：

1. **在人工系統中建模類人偏見**
2. **質疑客觀學習**與自我驅動學習的對比
3. **探索適應性與一致性之間的平衡**
4. **研究確認偏誤**既作為限制又作為特徵的雙重性

## Dependencies / 依賴項

```
torch >= 2.0.0
```

Install with: `pip install torch`

## Running the Code / 運行代碼

```bash
# Run the object-oriented implementation
python Python/EgoBasedNeuralNetwork.py

# Run the original implementation
python Python/OriginalEgoNetwork.py
```

Both scripts will train an ego-based neural network and display the training progress and final accuracy.

兩個腳本都會訓練一個自我導向神經網路並顯示訓練進度和最終準確率。