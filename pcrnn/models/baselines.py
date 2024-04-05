import torch
import torch.nn as nn


class NaiveForecaster:
    """Naive/persistent forecaster, always predicts previous value."""

    def __init__(self):
        pass
    
    def predict(self, X_test):
        y = X_test[:,-1]
        return y
    
    
class BaselineRNN(nn.Module):
    """Wrapper around RNNCell working as standard baseline RNN."""
    def __init__(self, layers):
                
        super().__init__()
        self.layers = layers
        self.alpha = 0

    def forward(self, input):
        output = self.layers(input)
        return output

    def net_u(self, inputs):
        Tb_pred = self(inputs)
        return Tb_pred
    
    def net_f(self, temp_pred, inputs):
        dT_pred = torch.tensor([0], dtype=torch.float).to(temp_pred.device)
        dT_comp = torch.tensor([0], dtype=torch.float).to(temp_pred.device)
        return dT_pred, dT_comp

    def predict(self, test_data):
        self.eval()
        with torch.no_grad():
            predictions = self(test_data)
        return predictions