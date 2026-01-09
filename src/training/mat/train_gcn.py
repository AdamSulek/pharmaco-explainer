import pickle
import logging
from torch_geometric.loader import DataLoader
from gcn_model import GCN
import torch.optim as optim
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import itertools
import pandas as pd
import argparse
import os
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(123)

lr_values = [0.001, 0.0001, 0.00001]  
batch_sizes = [16, 32, 64]  
conv_layers = [3, 4]  
model_dims = [128, 256, 512]  
dropout_rate = [0.0, 0.1, 0.2, 0.5] 
fc_hidden_dims = [64, 128, 256] 
num_fc_layers = [1, 2, 3]  

param_grid = list(itertools.product(lr_values, batch_sizes, conv_layers, model_dims, dropout_rate, fc_hidden_dims, num_fc_layers))

def calculate_metrics(labels, predictions, threshold=0.5):
    predictions = (np.array(predictions) > threshold).astype(int)
    
    accuracy = accuracy_score(labels, predictions)
    roc_auc = roc_auc_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)

    TP = ((np.array(labels) == 1) & (np.array(predictions) == 1)).sum()
    FP = ((np.array(labels) == 0) & (np.array(predictions) == 1)).sum()
    TN = ((np.array(labels) == 0) & (np.array(predictions) == 0)).sum()
    FN = ((np.array(labels) == 1) & (np.array(predictions) == 0)).sum()

    return accuracy, roc_auc, precision, recall, TP, FP, TN, FN

# Optimized train function
def train(model, train_loader, optimizer, criterion, threshold=0.5, label_param='label', device='cuda'):
    model.train()
    total_loss = 0
    all_labels = []
    all_predictions = []
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze(-1)
        
        # Dynamically get the correct label based on label_param
        labels = getattr(data, label_param).float()

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Get probabilities after sigmoid
        probabilities = torch.sigmoid(out)
        
        # Store labels and predictions
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(probabilities.detach().cpu().numpy())
    
    # Calculate metrics after the epoch
    avg_loss = total_loss / len(train_loader)
    accuracy, roc_auc, precision, recall, TP, FP, TN, FN = calculate_metrics(all_labels, all_predictions, threshold)

    return avg_loss, accuracy, roc_auc, precision, recall, TP, FP, TN, FN

# Optimized test function
def test(model, loader, threshold=0.5, label_param='label', device='cuda'):
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()
            
            # Dynamically get the correct label based on label_param
            labels = getattr(data, label_param).float()

            # Get probabilities after sigmoid
            probabilities = torch.sigmoid(out)
            
            # Store labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(probabilities.detach().cpu().numpy().flatten())
    
    # Calculate metrics after the evaluation
    accuracy, roc_auc, precision, recall, TP, FP, TN, FN = calculate_metrics(all_labels, all_predictions, threshold)

    return accuracy, roc_auc, precision, recall, TP, FP, TN, FN


def identify_influential_nodes(model, data, top_k=5):
    model.eval()
    data = data.to(device)
    
    output = model(data)
    
    model.zero_grad() 
    output.backward()
    
    gradients = model.final_conv_grads  
    node_activations = model.final_conv_acts
   
    node_importance = (gradients * node_activations).sum(dim=1)  
    
    _, top_indices = torch.topk(node_importance, top_k)
    top_nodes = top_indices.cpu().numpy()  
    
    return top_nodes, node_importance.cpu().detach().numpy()


best_val_roc = 0.0  
best_model_state = None  
best_params = None

gcn_architecture = []

def calculate_pos_weight(train_data, label_param):
    if label_param == "y":
        train_labels = [e.y for e in train_data]  # Retrieve e.y
    elif label_param == "activity":
        train_labels = [e.activity for e in train_data]  # Retrieve e.activity
    elif label_param == "label":
        train_labels = [e.label for e in train_data]
    else:
        raise ValueError("Invalid label_param")

    num_positive = sum(train_labels)
    num_negative = len(train_labels) - num_positive
    pos_weight = num_negative / num_positive
    return pos_weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script demonstrating argparse with a boolean flag.")
    parser.add_argument(
        "--concat_conv_layers",
        type=int,
        default=1,
        help="Enable or disable concatenation (default: True)"
    )
    #parser.add_argument("--dataset", type=str, choices=["cdk4", "cdk6"], required=True)
    parser.add_argument("--label_param", type=str, choices=["y", "activity", "label"], 
                        default='y')#required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--split_file_path", type=str, required=True)
    parser.add_argument("--split_type", type=str, choices=["split", "split_distant_set", "split_close_set"],
                        default='split')
    args = parser.parse_args()
    
    with open(f'{args.input_dir}/train.p', 'rb') as f:
        train_data_raw = pickle.load(f)
    
    with open(f'{args.input_dir}/val.p', 'rb') as f:
        val_data_raw = pickle.load(f)
        
    with open(f'{args.input_dir}/test.p', 'rb') as f:
        test_data = pickle.load(f)
        
    input_split_file = pd.read_parquet(f'{args.split_file_path}')[[ 'ID', f'{args.split_type}' ]]

    train_ids = input_split_file[input_split_file[f'{args.split_type}'] == 'train']['ID'].tolist()

    val_ids = input_split_file[input_split_file[f'{args.split_type}'].isin(['val', 'valid'])]['ID'].tolist()
    train_data = [data for data in train_data_raw if data.id in train_ids]
    val_data = [data for data in val_data_raw if data.id in val_ids]
    
    pos_weight = calculate_pos_weight(train_data, args.label_param)

    checkpoint_dir = f'{args.checkpoint_dir}/{args.k}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    result_dir = f'{args.result_dir}/{args.k}'
    os.makedirs(result_dir, exist_ok=True)

    best_val_roc = float('-inf')  # Ensure it starts from the lowest value
    best_hyperparams = None  # Track the best hyperparameters
    gcn_architecture = []

    for lr, batch_size, n_gcn_layers, model_dim, dropout_rate, fc_hidden_dim, num_fc_layers in param_grid:
        logging.info(f"Training model with lr={lr}, batch_size={batch_size}, n_layers={n_gcn_layers}, "
                    f"model_dim={model_dim}, fc_hidden_dim={fc_hidden_dim}, num_fc_layers={num_fc_layers}, "
                    f"concat_conv_layers={args.concat_conv_layers}, dropout_rate={dropout_rate}")

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        model = GCN(
            input_dim=42,
            model_dim=model_dim,
            concat_conv_layers=args.concat_conv_layers,
            n_layers=n_gcn_layers,
            dropout_rate=dropout_rate,
            fc_hidden_dim=fc_hidden_dim,
            num_fc_layers=num_fc_layers
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

        patience = 15
        epochs_without_improvement = 0

        for epoch in range(50):
            train_loss, train_acc, train_roc_auc, *_ = train(model, train_loader, optimizer, criterion, threshold=0.5, label_param=args.label_param)
            val_acc, val_roc_auc, *_ = test(model, val_loader, label_param=args.label_param)

            logging.info(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Accuracy {train_acc:.4f}, Train ROC AUC {train_roc_auc:.4f}')
            logging.info(f'Val Accuracy {val_acc:.4f}, Val ROC AUC {val_roc_auc:.4f}')

            if val_roc_auc > best_val_roc:
                best_val_roc = val_roc_auc
                best_model_path = os.path.join(checkpoint_dir, f'best_model_{args.split_type}.pth')
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Save Best Hyperparameters
                best_hyperparams = {
                    'lr': lr,
                    'batch_size': batch_size,
                    'n_layers': n_gcn_layers,
                    'model_dim': model_dim,
                    'fc_hidden_dim': fc_hidden_dim,
                    'num_fc_layers': num_fc_layers,
                    'concat_conv_layers': args.concat_conv_layers,
                    'dropout_rate': dropout_rate
                }

                # Save Model with Hyperparameters
                torch.save({
                    'state_dict': model.state_dict(),
                    'hyperparams': best_hyperparams
                }, best_model_path)

                logging.info(f'Saved best model with validation ROC AUC: {val_roc_auc:.4f}')
                epochs_without_improvement = 0    
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logging.info(f'Early stopping at epoch {epoch+1} due to no improvement in validation ROC AUC.')
                break

        gcn_architecture.append({
            **best_hyperparams,  # Add best hyperparameters to the architecture list
            "val_roc_auc": best_val_roc,
            "model_path": best_model_path
        })

    #df_architecture = pd.DataFrame(gcn_architecture)
    #df_architecture.to_csv(os.path.join(result_dir, 'best_architecture.csv'), index=False)

    if best_hyperparams:
        logging.info("Best Hyperparameters Found:")
        for key, value in best_hyperparams.items():
            logging.info(f"{key}: {value}")

    best_model_entry = max(gcn_architecture, key=lambda x: x["val_roc_auc"])
    best_model_path = best_model_entry["model_path"]

    if best_model_path:
        logging.info(f"Loading best model from: {best_model_path}")

        checkpoint = torch.load(best_model_path)
        hyperparams = checkpoint["hyperparams"]

        best_model = GCN(
            input_dim=42,
            model_dim=hyperparams["model_dim"],
            concat_conv_layers=hyperparams["concat_conv_layers"],
            n_layers=hyperparams["n_layers"],
            dropout_rate=hyperparams["dropout_rate"],
            fc_hidden_dim=hyperparams["fc_hidden_dim"],
            num_fc_layers=hyperparams["num_fc_layers"], 
            use_hooks=True
        ).to(device)

        best_model.load_state_dict(checkpoint["state_dict"])
        best_model.eval()

        test_acc, test_roc_auc, *_ = test(best_model, test_loader)
        
        logging.info(f'Test Accuracy: {test_acc:.4f}, Test ROC AUC: {test_roc_auc:.4f}')
    else:
        logging.info("No valid TEST model was found!")

        
    logging.info("Training completed!")
