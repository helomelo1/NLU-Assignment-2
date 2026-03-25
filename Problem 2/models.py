import torch
import torch.nn as nn

class VanillaRNN(nn.Module):
    """
    Trainable Parameters:

    W_xh : (input_size, hidden_size)
    W_hh : (hidden_size, hidden_size)
    b_h  : (hidden_size)

    W_hy : (hidden_size, output_size)
    b_y  : (output_size)

    Total Parameters:
    = input_size * hidden_size
    + hidden_size * hidden_size
    + hidden_size
    + hidden_size * output_size
    + output_size
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Weights to represent the current state input
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01) 

        # Weights to represent the previous state input
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)

        # Current state bias
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # Weights to convert final output (y) to representations
        self.W_hy = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)

        # Bias for the same
        self.b_y = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initialising state vectors for each state
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Computing every h
        for t in range(seq_len):
            h = torch.tanh(x[:, t] @ self.W_xh + h @ self.W_hh + self.b_h)

        out = h @ self.W_hy + self.b_y

        return out
    

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialising Traininable parameters
        self.W = nn.Parameter(torch.randn(input_size + hidden_size, hidden_size * 4) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_size * 4))

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = combined @ self.W + self.b

        f, i, o, g = gates.chunk(4, dim=1) # Getting the values of all the four gates
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    

class BiLSTM(nn.Module):
    """
    For ONE LSTM direction:

    W : (input_size + hidden_size, 4 * hidden_size)
    b : (4 * hidden_size)

    Reason: 4 gates → forget, input, output, candidate

    Total (one direction):
    = 4 * [(input_size + hidden_size) * hidden_size + hidden_size]

    Since BLSTM has TWO directions:

    Total LSTM params:
    = 2 * 4 * [(input_size + hidden_size) * hidden_size + hidden_size]

    Final Linear Layer:
    W_fc : (2 * hidden_size, output_size)
    b_fc : (output_size)

    Total Parameters:
    = 2 * 4 * [(input_size + hidden_size) * hidden_size + hidden_size]
    + (2 * hidden_size * output_size)
    + output_size
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Initialising LSTM Cells for dual way training i.e. forward and backward
        self.fwd = LSTMCell(input_size, hidden_size)
        self.bwd = LSTMCell(input_size, hidden_size)

        # Initialising a fully connected layer
        self.fc = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initialising state vectors for each state
        h_f, c_f = torch.zeros(batch_size, self.fwd.hidden_size, device=x.device), torch.zeros(batch_size, self.fwd.hidden_size, device=x.device)
        h_b, c_b = torch.zeros(batch_size, self.bwd.hidden_size, device=x.device), torch.zeros(batch_size, self.bwd.hidden_size, device=x.device)

        output_f, output_b = [], []

        # Forward
        for t in range(seq_len):
            h_f, c_f = self.fwd(x[:, t], h_f, c_f)
            output_f.append(h_f)

        # Backward
        for t in reversed(range(seq_len)):
            h_b, c_b = self.bwd(x[:, t], h_b, c_b)
            output_b.append(h_b)

        # Concatenating final states of forward and backward
        h = torch.cat([output_f[-1], output_b[-1]], dim=1)
        
        # Return output of the concat vector after passing through fully connected layer
        return self.fc(h)
    

class RNNAttention(nn.Module):
    """
    RNN Part:

    W_xh : (input_size, hidden_size)
    W_hh : (hidden_size, hidden_size)
    b_h  : (hidden_size)

    Attention Part:

    W_a : (hidden_size, 1)

    Output Layer:

    W_fc : (hidden_size, output_size)
    b_fc : (output_size)

    Total Parameters:
    = input_size * hidden_size
    + hidden_size * hidden_size
    + hidden_size
    + hidden_size * 1
    + hidden_size * output_size
    + output_size
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Weights to represent the current state input (Same as Vanilla RNN)
        self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01) 

        # Weights to represent the previous state input (Same as Vanilla RNN)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)

        # Current state bias (Same as Vanilla RNN)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

        # THE PART THAT IS DIFFERENT FROM THE VANILLA RNN IN ATTENTION RNN
        # Attention Weights
        self.W_a = nn.Parameter(torch.randn(hidden_size, 1))

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        hidden_states = []

        # Computing Hidden States
        for t in range(seq_len):
            h = torch.tanh(x[:, t] @ self.W_xh + h @ self.W_hh + self.b_h)
            hidden_states.append(h)

        H = torch.stack(hidden_states, dim=1) # (B, T, H)

        scores = torch.matmul(H, self.W_a).squeeze(-1) # (B, T)
        alpha = torch.softmax(scores, dim=1)

        context = torch.sum(H * alpha.unsqueeze(-1), dim=1)

        return self.fc(context)