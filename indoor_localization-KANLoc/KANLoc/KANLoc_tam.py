import torch
import torch.nn.functional as F
import math
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

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
import matplotlib
from sklearn.preprocessing import scale
from timeit import default_timer as timer
#from kan import KAN  # Adjust the import according to the actual file location
from sklearn.metrics import classification_report, precision_recall_fscore_support
# Constants
INPUT_DIM = 992
OUTPUT_DIM = 5  # Adjust for floor classification
VERBOSE = 1

def smooth(scalars, weight=0.6):
    smoothed = []
    last = scalars[0]
    for point in scalars:
        last = last * weight + (1 - weight) * point
        smoothed.append(last)
    return smoothed
'''def plot_loss_curve(train_losses, val_losses, output_path):
    plt.figure(figsize=(10, 7))
    matplotlib.rcParams['axes.linewidth'] = 2  # 设置边框线宽为2
    # 平滑处理
    smoothed_train = smooth(train_losses, weight=0.8)
    smoothed_val = smooth(val_losses, weight=0.8)
    # 绘制曲线
    plt.plot(smoothed_train, label='Training Loss', linewidth=2, color='blue', alpha=0.8)
    plt.plot(smoothed_val, label='Validation Loss', linewidth=2, color='orange', alpha=0.8)
    # 设置坐标轴
    plt.xlabel('Epochs', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    #plt.yscale('log')  # 对数刻度
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend(fontsize=22)
    plt.tight_layout()
    plt.savefig(output_path, dpi=500)
    plt.show()'''
# Parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-G", "--gpu_id", help="GPU device ID to run this script; default is 0; set to a negative number to use CPU", default=0, type=int)
    parser.add_argument("-R", "--random_seed", help="Random seed", default=0, type=int)
    parser.add_argument("-E", "--epochs", help="Number of training epochs; default is 200", default=300, type=int)
    #parser.add_argument("-E", "--epochs", help="Number of training epochs; default is 200", default=400, type=int)
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

    # Remap floor labels to be in range [0, OUTPUT_DIM-1]
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

    model = KAN(input_features=INPUT_DIM, layers_hidden=[512,256], grid_size=2, spline_order=2, grid_eps=0.02, dropout_prob=0).to(device)
    #model = KAN(input_features=INPUT_DIM, layers_hidden=[512,256], grid_size=2, spline_order=2, grid_eps=0.02, dropout_prob=0).to(device)
    criterion_coords = nn.L1Loss()
    criterion_floors = nn.CrossEntropyLoss()
    #weight_decay = 0.00001
    weight_decay = 0.00005
    optimizer = optim.AdamW(model.parameters(), lr=0.06, weight_decay=weight_decay)
    best_train_loss = float('inf')
    patience = 10
    trigger_times = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(x_train) // batch_size,
                                            #epochs=200)
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=0)

    train_losses = []
    val_losses = []
    start_time = timer()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs_coords, outputs_floors = model(x_train)
        loss_coords = criterion_coords(outputs_coords, y_train_coords)
        loss_floors = criterion_floors(outputs_floors, y_train_floors)
        #reg_loss = model.regularization_loss()
        loss = loss_coords + loss_floors#+reg_loss
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        # 早停机制
        '''if loss < best_train_loss:
            best_train_loss = loss
            trigger_times = 0
            # 保存最佳模型
            now = datetime.datetime.now()
            best_model_path=f'../results/model_pth/KANLoc_tam_new_model_{now.strftime("%Y%m%d-%H%M%S")}'
            torch.save(model.state_dict(),best_model_path )
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping at epoch {epoch}')
                break'''
        '''model.eval()
        with torch.no_grad():
            val_outputs_coords, val_outputs_floors = model(x_test)
            val_loss_coords = criterion_coords(val_outputs_coords, y_test_coords)
            val_loss_floors = criterion_floors(val_outputs_floors, y_test_floors)
            #reg_loss = model.regularization_loss()
            val_loss = val_loss_coords + val_loss_floors#+reg_loss

        scheduler.step(val_loss)'''
        train_losses.append(loss.item())
        #val_losses.append(val_loss.item())
        #print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

    end_time = timer()
    elapsed_time = end_time - start_time
    print(f'Training completed in {elapsed_time:.2f} seconds')


    # Save model
    now = datetime.datetime.now()
    best_model_path = f'../results/model_pth/KANLoc_tam_new_model_{now.strftime("%Y%m%d-%H%M%S")}'
    torch.save(model.state_dict(), best_model_path)
    #torch.save(model.state_dict(), f'../results/model_pth/KANLoc_tam_model_{now.strftime("%Y%m%d-%H%M%S")}.pth')
    #plot_loss_curve(train_losses, val_losses, f'dd_training_validation_loss_curve_{now.strftime("%Y%m%d-%H%M%S")}.png')
    #best_model_path=f'../results/model_pth/KANLoc_tam_model_20250305-163743.pth'
    model.load_state_dict(torch.load(best_model_path))
    # Evaluate model
    model.eval()
    with torch.no_grad():
        preds_coords, preds_floors = model(x_test)
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

    # Filter correct floor predictions
    correct_floor_mask = (y_test_floors == torch.argmax(preds_floors, dim=1))
    correct_pred_coords = preds_coords[correct_floor_mask]
    correct_test_coords = y_test_coords[correct_floor_mask]

    # Compute location error only in 2D (longitude and latitude) for correct floor predictions
    pos_err = np.sqrt(np.sum((correct_test_coords.cpu().numpy() - correct_pred_coords.cpu().numpy()) ** 2, axis=1)).mean()
    print(f'Location error (only correct floors): {pos_err} meters')

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

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 生成混淆矩阵数据
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    ax=sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=['F0', 'F1', 'F2', 'F3', 'F4'],
                yticklabels=['F0', 'F1', 'F2', 'F3', 'F4'],
                annot_kws={"fontsize": 12},
                   cbar=False,
                   )
    plt.xlabel('Predicted Floor', fontsize=16)
    plt.ylabel('True Floor', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # 调整颜色条字体大小
    cbar = plt.colorbar(ax.collections[0])# 传递 mappable 对象
    #cbar.set_label('label',fontsize=14)  # 颜色条标签字体大小
    cbar.ax.tick_params(labelsize=14)  # 颜色条刻度字体大小
    # 添加黑色边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)

    #plt.title('Floor Confusion Matrix')
    now = datetime.datetime.now()
    #plt.savefig(f'../results/kanloc_tam_confusion_matrix_{now.strftime("%Y%m%d-%H%M%S")}.png',
    #          dpi=500, bbox_inches='tight')
    plt.tight_layout()
    #plt.show()




    # 定义临时前向传播方法获取特征
    def modified_forward(self, x, update_grid=False):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, KANLinear) and update_grid:
                layer.update_grid(x)
            x = layer(x)
            if isinstance(layer, KANLinear):
                x = self.batch_norms[i // 2](x)
        features = x  # 最后一个隐藏层的输出
        coords_output = self.output_layer_coords(features)
        floors_output = self.output_layer_floors(features)
        return coords_output, floors_output, features


    # 保存原始前向传播方法
    original_forward = KAN.forward
    KAN.forward = modified_forward

    # 获取特征和预测结果
    model.eval()
    with torch.no_grad():
        _, _, features = model(x_test)
        features = features.cpu().numpy()

    # 恢复原始前向传播方法
    KAN.forward = original_forward

    # 准备标签数据
    y_test_floors_np = y_test_floors.cpu().numpy()

    # 随机采样部分数据（加快计算）
    np.random.seed(42)
    sample_idx = np.random.choice(len(features), len(features), replace=False)
    sampled_features = features[sample_idx]
    sampled_floors = y_test_floors_np[sample_idx]

    # 执行t-SNE降维
    #tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    tsne = TSNE(n_components=2, perplexity=150, n_iter=5000, learning_rate=200, init='random', random_state=42)
    features_tsne = tsne.fit_transform(sampled_features)

    # 创建可视化图表
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1],
                          c=sampled_floors, cmap='viridis', alpha=0.99,
                          s=20, edgecolor='none')
    #plt.title('t-SNE Visualization of Feature Space by Floor Level')
    plt.xlabel('tSNE_1',fontsize=15)
    plt.ylabel('tSNE_2',fontsize=15)
    cbar = plt.colorbar(scatter, ticks=np.unique(sampled_floors))
    cbar.set_label('Floor Level',fontsize=15)
    cbar.ax.tick_params(labelsize=15)
    # 添加图例
    handles = []
    for flr in np.unique(sampled_floors):
        handles.append(plt.scatter([], [], color=scatter.cmap(scatter.norm(flr)), label=f'Floor {flr}'))
    plt.legend(handles=handles, title='Floor Legend', title_fontsize=15, fontsize=15)
    plt.tight_layout()
    now = datetime.datetime.now()
    #plt.savefig(f'../results/kanloc_tam_tsne_visualization_{now.strftime("%Y%m%d-%H%M%S")}.png',dpi=500)
    #plt.show()


    # Save results
    now = datetime.datetime.now()
    '''path_out = f'../results/dd{now.strftime("%Y%m%d-%H%M%S")}.org'
    path_out1 = f'../results/dd_CM{now.strftime("%Y%m%d-%H%M%S")}.org'
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
        f.write(f"  - Location Error (meters, only correct floors): {pos_err}\n")'''