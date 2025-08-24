"""
Original Ego-Based Neural Network Implementation
原始自我導向神經網路實現

This is the original implementation as provided in the issue,
translated into a well-structured format while maintaining the same logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-Layer Perceptron as defined in the original issue."""
    
    def __init__(self, d, k): 
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, 128), 
            nn.ReLU(), 
            nn.Linear(128, k)
        )
    
    def forward(self, x): 
        return self.f(x)


def train_ego_based_neural_network():
    """
    Train ego-based neural network with the exact implementation from the issue.
    訓練自我導向神經網路，使用問題中的確切實現。
    """
    print("=== Original Ego-Based Neural Network ===")
    print("=== 原始自我導向神經網路 ===")
    print()
    
    # Toy data
    N, d, k = 512, 20, 3
    x = torch.randn(N, d)
    y = torch.randint(0, k, (N,))
    
    print(f"Dataset: {N} samples, {d} features, {k} classes")
    print(f"數據集：{N} 個樣本，{d} 個特徵，{k} 個類別")
    
    # Create model and store initial parameters (初始自我)
    model = MLP(d, k)
    theta0 = [p.detach().clone() for p in model.parameters()]  # 初始自我
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    
    # Ego strength parameters (ego 強度)
    lambda_p, lambda_o, alpha = 5e-3, 1e-2, 0.7
    
    print(f"\nHyperparameters:")
    print(f"λ_param: {lambda_p} (parameter inertia / 參數慣性)")
    print(f"λ_output: {lambda_o} (output inertia / 輸出慣性)")
    print(f"α: {alpha} (confirmation bias / 確認偏誤)")
    print(f"\nStarting training...")
    print("開始訓練...")
    
    prev_logits = None
    loss_history = []
    
    for epoch in range(200):
        opt.zero_grad()
        logits = model(x)
        
        # Task loss with confirmation bias weighting (任務損失帶確認偏誤權重)
        with torch.no_grad():
            if prev_logits is None:
                w = torch.ones_like(y, dtype=torch.float)
            else:
                agree = (prev_logits.argmax(1) == y)
                w = torch.where(
                    agree, 
                    torch.full_like(agree, alpha, dtype=torch.float), 
                    torch.full_like(agree, 1-alpha, dtype=torch.float)
                )
                w = w.float()
        
        task_loss = (F.cross_entropy(logits, y, reduction='none') * w).mean()
        
        # Parameter inertia (參數慣性)
        param_loss = 0.0
        for p, p0 in zip(model.parameters(), theta0):
            param_loss = param_loss + (p - p0).pow(2).sum()
        param_loss = lambda_p * param_loss
        
        # Output inertia (輸出慣性與上輪一致)
        if prev_logits is None:
            output_loss = torch.tensor(0., requires_grad=True)
        else:
            output_loss = lambda_o * F.mse_loss(logits, prev_logits)
        
        # Total loss
        loss = task_loss + param_loss + output_loss
        loss.backward()
        opt.step()
        
        # Update previous logits for next iteration
        prev_logits = logits.detach()
        
        # Store loss history
        loss_history.append({
            'epoch': epoch + 1,
            'total_loss': loss.item(),
            'task_loss': task_loss.item(),
            'param_loss': param_loss.item(),
            'output_loss': output_loss.item()
        })
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/200:")
            print(f"  Total Loss: {loss.item():.6f}")
            print(f"  Task Loss: {task_loss.item():.6f}")
            print(f"  Parameter Loss: {param_loss.item():.6f}")
            print(f"  Output Loss: {output_loss.item():.6f}")
    
    print("\nTraining completed!")
    print("訓練完成！")
    
    # Evaluate final performance
    with torch.no_grad():
        final_logits = model(x)
        final_predictions = final_logits.argmax(dim=1)
        accuracy = (final_predictions == y).float().mean()
        print(f"\nFinal Accuracy: {accuracy:.4f}")
        print(f"最終準確率: {accuracy:.4f}")
    
    return model, loss_history


if __name__ == "__main__":
    model, history = train_ego_based_neural_network()
    print("done")