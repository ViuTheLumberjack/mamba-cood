import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# ConvLSTM Cell
# -------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, x, h_prev, c_prev):
        # x:     [B, C, H, W]
        # h_prev, c_prev: [B, hidden_dim, H, W]

        combined = torch.cat([x, h_prev], dim=1)  # [B, C+hidden_dim, H, W]
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

# -------------------------------
# ConvLSTM Layer (unrolled)
# -------------------------------
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size))

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()
        h, c = [None] * len(self.layers), [None] * len(self.layers)

        for t in range(T):
            input_t = x[:, t]  # [B, C, H, W]
            for i, layer in enumerate(self.layers):
                if h[i] is None:
                    h[i] = torch.zeros(B, layer.hidden_dim, H, W, device=x.device)
                    c[i] = torch.zeros(B, layer.hidden_dim, H, W, device=x.device)
                h[i], c[i] = layer(input_t, h[i], c[i])
                input_t = h[i]

        return h[-1]  # Return the hidden state from the last layer at last time step

# -------------------------------
# Full Model: Past → Future Frame
# -------------------------------
class FutureFramePredictor(nn.Module):
    def __init__(self, input_channels=8, hidden_dim=32, kernel_size=(3, 3)):
        super().__init__()
        self.encoder = ConvLSTM(input_dim=input_channels, hidden_dim=hidden_dim, kernel_size=kernel_size)
        self.decoder = nn.Conv2d(hidden_dim, input_channels, kernel_size=1)  # Project back to 8 channels

    def forward(self, x):
        # x: [B, 3, 8, 48, 176]
        h = self.encoder(x)  # [B, hidden_dim, H, W]
        out = self.decoder(h)  # [B, 8, H, W]
        return out#.unsqueeze(1)  # [B, 1, 8, H, W]
    

if __name__ == "__main__":
    # Define model
    model = FutureFramePredictor(input_channels=8, hidden_dim=32, kernel_size=(3, 3))
    model.eval()  # Set model to evaluation mode

    # Create a random input tensor: [B, T, C, H, W]
    B, T, C, H, W = 2, 3, 8, 48, 176  # Example batch size and spatial dimensions
    input_tensor = torch.randn(B, T, C, H, W)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Print output shape
    print("Output shape:", output.shape)  # Expected: [B, 1, 8, H, W]
