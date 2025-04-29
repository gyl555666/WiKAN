import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_features, layers_hidden, output_features):
        super(MLP, self).__init__()
        layers = []
        prev_features = input_features
        for hidden_features in layers_hidden:
            layers.append(nn.Linear(prev_features, hidden_features))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout_prob))
            prev_features = hidden_features
        layers.append(nn.Linear(prev_features, output_features))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    def _initialize_weights(self):
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
    def forward(self, x):
        return self.network(x)

