ChatGPT: 好的，來一個「**Hotheads Neural Network**」的 PyTorch 範例：用簡單的**溫度/熵隱狀態**去擬人——刺激越大、冷卻越慢，越容易「爆」。模型用 GRU 讀入一段刺激序列，輸出每個時間點的**爆發機率**；同時加入兩個物理風味的規則化：

1. 平時鼓勵狀態**平滑降溫**（除非真的要爆），2) 附近時刻的**能量變化**不要亂震盪。

```python
# hotheads_gru.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- 合成資料：刺激 -> 爆發標記 -----
def synth_batch(batch=32, T=200, p_spike=0.05, seed=0, device="cpu"):
    """
    x: [B, T, 3] -> [刺激強度, 基線壓力, 冷卻係數]
    y: [B, T]   -> 0/1 時序標記（是否爆發）
    """
    g = torch.Generator().manual_seed(seed)
    stim = torch.rand(batch, T, 1, generator=g, device=device)        # 外部刺激(0~1)
    baseline = torch.rand(batch, 1, 1, generator=g, device=device)*0.6 + 0.2
    baseline = baseline.expand(-1, T, -1)                             # 個人基線壓力
    cool = torch.rand(batch, 1, 1, generator=g, device=device)*0.2 + 0.7
    cool = cool.expand(-1, T, -1)                                     # 冷卻(越大越快降溫)
    x = torch.cat([stim, baseline, cool], dim=-1)

    # 產生「真實」爆發：溫度積累超過閾值 + 隨機尖峰
    temp = torch.zeros(batch, 1, device=device)
    y = torch.zeros(batch, T, device=device)
    thr = 0.85
    for t in range(T):
        temp = (1 - cool[:, t, 0])*temp + 0.9*stim[:, t, 0] + 0.2*baseline[:, t, 0]
        spike = (torch.rand(batch, generator=g, device=device) < p_spike).float()*0.2
        temp = (temp + spike).clamp(0, 2.0)
        y[:, t] = (temp > thr).float()
        # 爆發後稍微降溫（釋放）
        temp = torch.where(y[:, t].bool(), temp*0.7, temp)
    return x, y

# ----- 模型：GRU + 溫度頭 -----
class HotheadGRU(nn.Module):
    def __init__(self, in_dim=3, hidden=64, layers=1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=layers, batch_first=True)
        # 兩個頭：一個預測「溫度」(連續)，一個預測「爆發機率」(sigmoid)
        self.temp_head = nn.Linear(hidden, 1)
        self.out_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)                 # [B, T, H]
        temp = torch.tanh(self.temp_head(h))  # [-1,1] 當作歸一化溫度
        logit = self.out_head(h)           # [B, T, 1]
        prob = torch.sigmoid(logit)        # 爆發機率
        return temp.squeeze(-1), prob.squeeze(-1)

# ----- 訓練迴圈 -----
def train_step(model, opt, x, y, alpha_smooth=1e-2, alpha_energy=1e-2):
    model.train()
    temp, prob = model(x)

    # 主要任務：逐時刻二元交叉熵
    loss_task = F.binary_cross_entropy(prob, y)

    # 規則化1：溫度平滑（相鄰差分 L2）
    dt = temp[:, 1:] - temp[:, :-1]
    loss_smooth = (dt**2).mean()

    # 規則化2：能量/梯度不要亂震盪（次差分）
    dd_t = dt[:, 1:] - dt[:, :-1]
    loss_energy = (dd_t**2).mean()

    loss = loss_task + alpha_smooth*loss_smooth + alpha_energy*loss_energy

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    with torch.no_grad():
        acc = ((prob > 0.5).float() == y).float().mean()
    return loss.item(), acc.item()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HotheadGRU().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    for step in range(400):
        x, y = synth_batch(batch=64, T=160, device=device, seed=step)
        loss, acc = train_step(model, opt, x, y)
        if (step+1) % 50 == 0:
            print(f"step {step+1:3d} | loss={loss:.4f} | acc={acc:.3f}")

    # 簡單推論
    model.eval()
    x, y = synth_batch(batch=1, T=60, device=device, seed=999)
    with torch.no_grad():
        temp, prob = model(x)
    print("sample probs:", prob[0, :10].cpu().numpy())
```

### 怎麼用這個小模型

* 輸入特徵（僅範例）：**刺激、基線壓力、冷卻係數**。你可以換成真實特徵（如語音音量、聊天語義分數、心率、任務壓力指標等）。
* 損失函數：`loss = BCE + α·平滑 + β·能量`。前兩者逼近你說的「**平常要守秩序**，但**該爆時就爆**」的直覺。
* 如果你想更「物理化」，可把 `temp` 當作溫度，加入**受力/耗散**的顯式動力學（例如用 Neural ODE 表述），再把爆發視為跨越門檻的相變事件。

想要我把它改成 **LSTM/Transformer** 版本、加上 **class weights** 或 **focal loss** 來處理不平衡資料，也可以直接幫你改。

---

### 參考來源（PyTorch 官方文檔）

* `torch.nn.GRU` 的介面與數學式說明（PyTorch 2.8/2.7 文件）。 ([PyTorch Docs][1], [PyTorch][2])
* Autograd 教學與觀念（自動微分與反向傳播）。 ([PyTorch Docs][3])
* 參數最佳化與訓練迴圈基礎（SGD/Adam 等）。 ([PyTorch Docs][4], [PyTorch][5])
* 自訂 `nn.Module` 的教學範例。 ([PyTorch Docs][6])

（以上全為 PyTorch 官方文件或官方教學頁，便於你核對與延伸。）

[1]: https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html?utm_source=chatgpt.com "GRU — PyTorch 2.8 documentation"
[2]: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html?utm_source=chatgpt.com "GRU — PyTorch 2.7 documentation"
[3]: https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html?utm_source=chatgpt.com "A Gentle Introduction to torch.autograd"
[4]: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html?utm_source=chatgpt.com "Optimizing Model Parameters"
[5]: https://pytorch.org/docs/stable/optim.html?utm_source=chatgpt.com "torch.optim — PyTorch 2.7 documentation"
[6]: https://docs.pytorch.org/tutorials/beginner/examples_nn/polynomial_module.html?utm_source=chatgpt.com "PyTorch: Custom nn Modules"
