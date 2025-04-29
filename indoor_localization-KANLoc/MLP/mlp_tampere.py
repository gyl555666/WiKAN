import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim_coords, output_dim_floors, dropout_prob=0):
        super(MLP, self).__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            input_dim = dim
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer_coords = nn.Linear(hidden_dims[-1], output_dim_coords)  # 经纬度输出
        self.output_layer_floors = nn.Linear(hidden_dims[-1], output_dim_floors)
        #self._initialize_weights()
    #def _initialize_weights(self):
    #    for m in self.network:
    #        if isinstance(m, nn.Linear):
    #            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    #            if m.bias is not None:
    #                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
    #                bound = 1 / math.sqrt(fan_in)
    #                nn.init.uniform_(m.bias, -bound, bound)
    def forward(self, x):
        x = self.hidden_layers(x)
        coords_output = self.output_layer_coords(x)
        floors_output = self.output_layer_floors(x)
        return coords_output, floors_output
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from timeit import default_timer as timer
#from mlp import MLP  # Adjust the import according to the actual file location
from sklearn.metrics import classification_report, precision_recall_fscore_support
# Constants
INPUT_DIM = 992
OUTPUT_DIM_COORDS = 2  # Adjust for coordinate output
OUTPUT_DIM_FLOORS = 5  # Adjust for floor classification
VERBOSE = 1

def plot_loss_curve(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

# Parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-G", "--gpu_id", help="GPU device ID to run this script; default is 0; set to a negative number to use CPU", default=0, type=int)
    parser.add_argument("-R", "--random_seed", help="Random seed", default=0, type=int)
    parser.add_argument("-E", "--epochs", help="Number of training epochs; default is 200", default=300, type=int)
    parser.add_argument("-B", "--batch_size", help="Batch size; default is 10", default=10, type=int)
    parser.add_argument("--dropout_prob", help="Dropout probability; default is 0.5", default=0.0, type=float)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    dropout_prob = args.dropout_prob

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(random_seed)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    # Load and preprocess data
    train_rss = pd.read_csv('../Tamperedata/Training_rss_21Aug17.csv', header=None)
    train_coords = pd.read_csv('../Tamperedata/Training_coordinates_21Aug17.csv', header=None)
    test_rss = pd.read_csv('../Tamperedata/Test_rss_21Aug17.csv', header=None)
    test_coords = pd.read_csv('../Tamperedata/Test_coordinates_21Aug17.csv', header=None)

    # Check shapes of loaded data
    print(f'Shape of train_rss: {train_rss.shape}')
    print(f'Shape of train_coords: {train_coords.shape}')
    print(f'Shape of test_rss: {test_rss.shape}')
    print(f'Shape of test_coords: {test_coords.shape}')

    # Ensure the data shapes are consistent
    assert train_rss.shape[0] == train_coords.shape[0], "train_rss和train_coords应该有相同的行数"
    assert test_rss.shape[0] == test_coords.shape[0], "test_rss和test_coords应该有相同的行数"

    train_rss = scale(train_rss, axis=1)
    test_rss = scale(test_rss, axis=1)

    # Extract floor information from coordinates (assuming floor is the third column)
    train_floors = train_coords.iloc[:, 2].astype(int)
    test_floors = test_coords.iloc[:, 2].astype(int)

    # Remap floor labels to be in range [0, OUTPUT_DIM_FLOORS-1]
    floor_mapping = {floor: idx for idx, floor in enumerate(np.unique(train_floors))}
    inverse_floor_mapping = {idx: floor for floor, idx in floor_mapping.items()}
    train_floors = train_floors.map(floor_mapping)
    test_floors = test_floors.map(floor_mapping)

    x_train = torch.tensor(train_rss, dtype=torch.float32).to(device)
    y_train_coords = torch.tensor(train_coords.iloc[:, :2].values, dtype=torch.float32).to(device)
    y_train_floors = torch.tensor(train_floors.values, dtype=torch.long).to(device)

    x_test = torch.tensor(test_rss, dtype=torch.float32).to(device)
    y_test_coords = torch.tensor(test_coords.iloc[:, :2].values, dtype=torch.float32).to(device)
    y_test_floors = torch.tensor(test_floors.values, dtype=torch.long).to(device)

    model = MLP(input_dim=INPUT_DIM, hidden_dims=[ 128, 64], output_dim_coords=OUTPUT_DIM_COORDS,output_dim_floors=OUTPUT_DIM_FLOORS, dropout_prob=0).to(device)
    #model = MLP(input_dim=INPUT_DIM, hidden_dims=[128,64], output_dim_coords=OUTPUT_DIM_COORDS, output_dim_floors=OUTPUT_DIM_FLOORS, dropout_prob=0).to(device)
    criterion_coords = nn.L1Loss()
    criterion_floors = nn.CrossEntropyLoss()
    weight_decay = 0.00005
    optimizer = optim.AdamW(model.parameters(), lr=0.06, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)

    train_losses = []
    val_losses = []
    start_time = timer()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs_coords, outputs_floors = model(x_train)
        loss_coords = criterion_coords(outputs_coords, y_train_coords)
        loss_floors = criterion_floors(outputs_floors, y_train_floors)
        loss = loss_coords + loss_floors
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs_coords, val_outputs_floors = model(x_test)
            val_loss_coords = criterion_coords(val_outputs_coords, y_test_coords)
            val_loss_floors = criterion_floors(val_outputs_floors, y_test_floors)
            val_loss = val_loss_coords + val_loss_floors

        scheduler.step(val_loss)
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

    end_time = timer()
    elapsed_time = end_time - start_time
    print(f'Training completed in {elapsed_time:.2f} seconds')

    plot_loss_curve(train_losses, val_losses, 'mlp_tampere_training_validation_loss_curve.png')

    # Save model
    now = datetime.datetime.now()
    torch.save(model.state_dict(), f'../results/pth/mlp_tampere_model_{now.strftime("%Y%m%d-%H%M%S")}.pth')

    # Evaluate model
    model.eval()
    with torch.no_grad():
        preds_coords, preds_floors = model(x_test)
    # Calculate floor classification accuracy
    print(f'Shape of y_test_floors: {y_test_floors.shape}')
    print(f'Shape of preds_floors: {preds_floors.shape}')
    print(f'First few y_test_floors: {y_test_floors[:5]}')
    print(f'First few preds_floors: {preds_floors[:5]}')

    floor_results = (y_test_floors == torch.argmax(preds_floors, dim=1)).float()
    acc_flr = floor_results.mean().item()
    print(f'Floor accuracy: {acc_flr}')

    # Convert predicted floors back to original labels
    predicted_floors = torch.argmax(preds_floors, dim=1).cpu().numpy()
    predicted_floors = np.vectorize(inverse_floor_mapping.get)(predicted_floors)

    # Compute location error only in 2D (longitude and latitude) for correctly classified floors
    correct_floor_indices = (y_test_floors == torch.argmax(preds_floors, dim=1)).cpu().numpy().astype(bool)
    pos_err = np.sqrt(np.sum((y_test_coords.cpu().numpy()[correct_floor_indices] - preds_coords.cpu().numpy()[correct_floor_indices]) ** 2, axis=1)).mean()
    print(f'Location error for correctly classified floors: {pos_err} meters')
    # Calculate classification metrics
    y_true = y_test_floors.cpu().numpy()
    y_pred = torch.argmax(preds_floors, dim=1).cpu().numpy()

    # 计算详细分类报告
    report = classification_report(y_true, y_pred, output_dict=True)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']

    # 打印指标
    print("\n=== Classification Metrics ===")
    print(f"macro Precision: {precision:.4f}")
    print(f"macro Recall: {recall:.4f}")
    print(f"macro F1 Score: {f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))
    # Save results
    now = datetime.datetime.now()
    path_out = f'../results/mlp_tampere{now.strftime("%Y%m%d-%H%M%S")}.org'
    path_out1 = f'../results/mlp_tampere_CM{now.strftime("%Y%m%d-%H%M%S")}.org'
    with open(path_out1, 'w') as f:
        f.write("* Classification Metrics\n")
        f.write(f"  - Macro Precision: {precision:.4f}\n")
        f.write(f"  - Macro Recall: {recall:.4f}\n")
        f.write(f"  - Macro F1 Score: {f1:.4f}\n")
        f.write("\n* Detailed Classification Report\n")
        f.write(classification_report(y_true, y_pred))
    with open(path_out, 'w') as f:
        f.write("#+STARTUP: showall\n")  # unfold everything when opening
        f.write("* System Parameters\n")
        f.write(f"  - Numpy Random Seed: {random_seed}\n")
        f.write(f"  - Number of Epochs: {epochs}\n")
        f.write(f"  - Batch Size: {batch_size}\n")
        f.write(f"  - Dropout Probability: {dropout_prob}\n")
        f.write(f"  - Hidden Layers: {[128, 64]}\n")
        f.write("* Performance\n")
        f.write(f"  - Floor Accuracy: {acc_flr}\n")
        f.write(f"  - Location Error (meters): {pos_err}\n")
