# A take away

Here are your **Dense (Feedforward) Network** and **Recurrent Neural Network (LSTM/GRU)** implementations, now **with jokes** to keep your debugging sessions fun! 😆💻🔥  

---

## **1. Dense Neural Network (a.k.a. the "Lazy Model")**
This one doesn't care about word order—just takes the average of word embeddings like someone **"winging it"** in an exam. 🤡  

### **Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout, pad_idx):
        super(DenseNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)  # First dense layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text shape: (batch_size, seq_len)
        embedded = self.embedding(text)  # (batch_size, seq_len, embedding_dim)
        pooled = torch.mean(embedded, dim=1)  # Just averaging like a last-minute group project
        hidden = F.relu(self.fc1(pooled))  # First dense layer with ReLU activation
        hidden = self.dropout(hidden)  # Apply dropout, because life is full of surprises
        output = self.fc2(hidden)  # Output layer, let's hope it's right!
        return output
```
---
### **Joke About Dense Networks** 🤖  
> _Why did the dense network fail the NLP test?_  
> Because it **ignored the context** and thought "break a leg" was a medical emergency. 🚑😂  

---

## **2. Recurrent Neural Network (a.k.a. the "Overthinker")**
This model actually **remembers** what it reads, unlike me after an all-nighter. 🧠💀  

### **Implementation:**
```python
class RecurrentNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout, pad_idx, rnn_type='LSTM'):
        super(RecurrentNetwork, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                               bidirectional=bidirectional, dropout=dropout, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, 
                              bidirectional=bidirectional, dropout=dropout, batch_first=True)
        else:
            raise ValueError("rnn_type must be either 'LSTM' or 'GRU'")

        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # Embedding layer
        rnn_output, _ = self.rnn(embedded)  # Pass through LSTM/GRU
        final_hidden_state = rnn_output[:, -1, :]  # Take last hidden state
        output = self.fc(final_hidden_state)  # Fully connected layer
        return output
```

---
### **Joke About RNNs** 🧠💾  
> _Why did the LSTM go to therapy?_  
> Because it **couldn’t forget its past** traumas! 😭😂  

---

## **Final Thoughts**
🚀 **Dense Networks:** "Let's just average everything and hope for the best."  
🧠 **Recurrent Networks:** "I'm remembering every detail… even the ones I shouldn't."  

---
Now, get that **A+** and show those neural networks who's boss! Let me know if you need help with **training code (`train_model()`)** or **hyperparameter tuning.** 😆🔥
