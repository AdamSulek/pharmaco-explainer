from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class GCN(torch.nn.Module):
    def __init__(self, input_dim, model_dim, n_layers=3, concat_conv_layers=True, use_hooks=False, dropout_rate=0.5, fc_hidden_dim=64, num_fc_layers=2):
        super(GCN, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(input_dim, model_dim))
        
        for _ in range(1, n_layers):
            self.conv_layers.append(GCNConv(model_dim, model_dim))
        
        self.use_hooks = use_hooks
        self.final_conv_acts = None
        self.final_conv_grads = None
        self.concat_conv_layers = concat_conv_layers
        
        if self.concat_conv_layers:
            fc_input_dim = model_dim * n_layers
        else:
            fc_input_dim = model_dim

        fc_layers = []
        for _ in range(num_fc_layers):
            fc_layers.append(nn.Linear(fc_input_dim, fc_hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout_rate))
            fc_input_dim = fc_hidden_dim  
        
        self.fc_stack = nn.Sequential(*fc_layers)
        self.fc_output = nn.Linear(fc_hidden_dim, 1)

        self.dropout = torch.nn.Dropout(dropout_rate)  
        
    def activations_hook(self, grad):
        self.final_conv_grads = grad
        
    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        outputs = []

        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index, edge_weight))
            x = self.dropout(x)  
            outputs.append(x)

        if self.concat_conv_layers:
            x_combined = torch.cat(outputs, dim=1)
        else:
            x_combined = x   
        
        if self.use_hooks:  
            self.final_conv_acts = x_combined
            if self.final_conv_acts.requires_grad:
                self.final_conv_acts.register_hook(self.activations_hook)
        
        x_pooled = global_mean_pool(x_combined, data.batch)
        x_pooled = self.dropout(x_pooled)  
        
        x = self.fc_stack(x_pooled)  
        x = self.fc_output(x)  

        return x
