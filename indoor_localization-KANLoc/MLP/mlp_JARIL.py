import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
# 设置 Matplotlib 的默认字体为一个支持中文的字体，例如 'SimHei'（黑体）
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import datetime
import math
class MLP(nn.Module):
    def __init__(self, input_features, layers_hidden, output_features=16):
        super(MLP, self).__init__()
        layers = []
        prev_features = input_features
        for hidden_features in layers_hidden:
            layers.append(nn.Linear(prev_features, hidden_features))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(dropout_prob))
            prev_features = hidden_features
        layers.append(nn.Linear(prev_features, output_features))
        #layers.append(nn.ReLU())
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
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import os
import math
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import scale
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.optim as optim

### 导入新的 MLP 模型
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os
import math
import numpy as np
import scipy.io as sio
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support
from timeit import default_timer as timer
# 位置标签到二维坐标的映射
def get_label_to_coord_mapping(num_labels=16):
    """
    将位置标签映射到二维坐标。
    根据用户指定的逻辑，将16个标签映射到4x4网格上，每个网格间距0.8米。

    Args:
        num_labels (int): 位置标签的数量。

    Returns:
        list of tuples: 每个标签对应的 (x, y) 坐标。
    """
    if num_labels != 16:
        raise ValueError("当前映射逻辑仅支持16个标签。请根据需要调整映射函数。")
    coords = np.array([
        [i // 4* 0.8, i % 4 * 0.8] for i in range(16)
    ])
    return coords.tolist()


# 计算平均定位误差 (ALE)
def compute_average_error(predictions, targets, label_to_coord):
    """
    计算平均定位误差 (ALE)，单位为米。

    Args:
        predictions (list or np.array): 预测的标签列表。
        targets (list or np.array): 真实的标签列表。
        label_to_coord (list of tuples): 标签到二维坐标的映射列表。

    Returns:
        float: 平均定位误差 (米)。
    """
    pred_coords = [label_to_coord[pred] for pred in predictions]
    true_coords = [label_to_coord[true] for true in targets]
    pred_coords = np.array(pred_coords)
    true_coords = np.array(true_coords)
    errors = np.linalg.norm(pred_coords - true_coords, axis=1)
    average_error = errors.mean()
    return average_error
def main():
    # 设置参数
    #batch_size = 128
    batch_size=256
    num_epochs = 600
    #learning_rate = 0.004
    learning_rate = 0.05
    #milestone_steps = [10,50, 100, 150]
    milestone_steps = [10,50,100,150]#学习率调整的里程碑
    #gamma = 0.9  # 学习率衰减因子
    gamma = 0.5  # 学习率衰减因子

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建结果保存目录
    os.makedirs('mlp_weights', exist_ok=True)
    os.makedirs('mlp_result', exist_ok=True)
    os.makedirs('mlp_vis', exist_ok=True)

    # 加载训练数据
    print("Loading training data...")
    train_data_amp = sio.loadmat('../csi_data/train_data_split_amp.mat')['train_data']
    train_location_label = sio.loadmat('../csi_data/train_data_split_amp.mat')['train_location_label']
    train_labels_combined = train_location_label.squeeze()
    num_train_instances = train_data_amp.shape[0]

    train_data = torch.from_numpy(train_data_amp).float()
    train_labels = torch.from_numpy(train_labels_combined).long()

    train_dataset = TensorDataset(train_data, train_labels)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 加载测试数据
    print("Loading test data...")
    test_data_amp = sio.loadmat('../csi_data/test_data_split_amp.mat')['test_data']
    test_location_label = sio.loadmat('../csi_data/test_data_split_amp.mat')['test_location_label']
    test_labels_combined = test_location_label.squeeze()
    num_test_instances = test_data_amp.shape[0]

    test_data = torch.from_numpy(test_data_amp).float()
    test_labels = torch.from_numpy(test_labels_combined).long()

    test_dataset = TensorDataset(test_data, test_labels)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 定义位置标签到二维坐标的映射
    num_labels = max(train_labels.max().item(), test_labels.max().item()) + 1
    if num_labels != 16:
        raise ValueError(f"当前标签数量为 {num_labels}，但映射函数仅支持16个标签。请根据需要调整映射函数。")
    label_to_coord = get_label_to_coord_mapping(num_labels=num_labels)

    train_data = torch.mean(train_data, dim=-1)[0]
    test_data = torch.mean(test_data, dim=-1)[0]
    #train_data = torch.mean(train_data, dim=-1)
    #test_data = torch.mean(test_data, dim=-1)
    # 初始化模型
    print("Initializing KAN model...")
    hidden_layers = [ 512, 256]  # 隐藏层配置，请根据需要调整
    mlp_model = MLP(input_features=52, layers_hidden=hidden_layers).cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone_steps, gamma=gamma)

    # 训练和评估记录
    train_loss_loc = np.zeros(num_epochs)
    test_loss_loc = np.zeros(num_epochs)
    train_acc_loc = np.zeros(num_epochs)
    test_acc_loc = np.zeros(num_epochs)
    test_avg_error = np.zeros(num_epochs)

    best_test_acc = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_model_path = ""  # 初始化保存路径
    # 开始训练
    start_time=timer()
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        mlp_model.train()


        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for samples, labels in tqdm(train_data_loader, desc='Training'):
            # 确保输入数据为 [batch_size, 52]
            if len(samples.shape) > 2:
                samples = samples.mean(dim=-1)  # 对时间维度进行池化
            samples = samples.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = mlp_model(samples)
            loss = criterion(outputs, labels)


            loss.backward()
            optimizer.step()

            running_loss += loss.item() * samples.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        scheduler.step()
        epoch_loss = running_loss / total_train
        epoch_acc = 100.0 * correct_train / total_train
        train_loss_loc[epoch] = epoch_loss
        train_acc_loc[epoch] = epoch_acc

        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%')
        end_time = timer()
        elapsed_time = end_time - start_time
        print(f'Training completed in {elapsed_time:.2f} seconds')
        # 测试阶段
        mlp_model.eval()
        running_loss_test = 0.0
        correct_test = 0
        total_test = 0
        all_test_predictions = []
        all_test_labels = []

        with torch.no_grad():
            for samples, labels in tqdm(test_data_loader, desc='Testing'):
                if len(samples.shape) > 2:
                    samples = samples.mean(dim=-1)  # 对时间维度进行池化
                samples = samples.cuda()
                labels = labels.cuda()

                outputs = mlp_model(samples)
                loss = criterion(outputs, labels)

                running_loss_test += loss.item() * samples.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_test += (predicted == labels).sum().item()
                total_test += labels.size(0)

                all_test_predictions.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())

        epoch_loss_test = running_loss_test / total_test
        epoch_acc_test = 100.0 * correct_test / total_test
        avg_error_test = compute_average_error(all_test_predictions, all_test_labels, label_to_coord)

        test_loss_loc[epoch] = epoch_loss_test
        test_acc_loc[epoch] = epoch_acc_test
        test_avg_error[epoch] = avg_error_test

        print(
            f'Test Loss: {epoch_loss_test:.4f}, Test Accuracy: {epoch_acc_test:.2f}%, Test Avg Error: {avg_error_test:.4f}m')
        print("\nClassification Report for Epoch {}:".format(epoch + 1))
        print(classification_report(all_test_labels, all_test_predictions, digits=4, zero_division=0))

        # 计算并打印宏平均
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_test_labels, all_test_predictions, average='macro', zero_division=0
        )

        print(f"Macro Average - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
        # 保存最佳模型
        if epoch_acc_test > best_test_acc:
            best_test_acc = epoch_acc_test
            best_precision = precision_macro
            best_recall = recall_macro
            best_f1 = f1_macro
            best_model_path = f'mlp_weights/mlp_model_best_epoch{epoch + 1}_Acc{epoch_acc_test:.3f}.pth'  # 添加路径记录
            torch.save(mlp_model.state_dict(), best_model_path)  # 取消注释
            print(f'Best model saved at epoch {epoch + 1} with Test Accuracy: {epoch_acc_test:.3f}%')
            #torch.save(mlp_model.state_dict(), f'mlp_weights/mlp_model_best_epoch{epoch + 1}_Acc{epoch_acc_test:.3f}.pth')
            #print(f'Best model saved at epoch {epoch + 1} with Test Accuracy: {epoch_acc_test:.3f}%')'''

    # 保存训练和测试结果
    print("Saving results...")
    sio.savemat(f'mlp_result/train_loss_loc_{timestamp}.mat', {'train_loss_loc': train_loss_loc})
    sio.savemat(f'mlp_result/test_loss_loc_{timestamp}.mat', {'test_loss_loc': test_loss_loc})
    sio.savemat(f'mlp_result/train_acc_loc_{timestamp}.mat', {'train_acc_loc': train_acc_loc})
    sio.savemat(f'mlp_result/test_acc_loc_{timestamp}.mat', {'test_acc_loc': test_acc_loc})
    sio.savemat(f'mlp_result/test_avg_error_{timestamp}.mat', {'test_avg_error': test_avg_error})
    sio.savemat(f'mlp_result/best_precision_{timestamp}.mat', {'best_precision': best_precision})
    sio.savemat(f'mlp_result/best_recall_{timestamp}.mat', {'best_recall': best_recall})
    sio.savemat(f'mlp_result/best_f1_{timestamp}.mat', {'best_f1': best_f1})
    print("Training and testing completed.")
    print("\n\n=== 使用最佳模型进行最终评估 ===")
    # 加载最佳模型
    mlp_model.load_state_dict(torch.load(best_model_path))
    mlp_model.eval()

    # 重新运行测试集预测
    all_test_predictions = []
    all_test_labels = []

    with torch.no_grad():
        for samples, labels in test_data_loader:
            if len(samples.shape) > 2:
                samples = samples.mean(dim=-1)
            samples = samples.cuda()
            labels = labels.cuda()

            outputs = mlp_model(samples)
            _, predicted = torch.max(outputs.data, 1)

            all_test_predictions.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    # 计算最终指标
    final_avg_error = compute_average_error(all_test_predictions, all_test_labels, label_to_coord)
    final_acc = 100.0 * np.mean(np.array(all_test_predictions) == np.array(all_test_labels))

    print(f"\n最佳模型最终测试准确率: {final_acc:.2f}%")
    print(f"最佳模型最终平均定位误差: {final_avg_error:.4f}m")
    # 最终评估
    print("\nFinal Evaluation on Test Set:")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Precision : {best_precision:.4f}")
    print(f"Recall : {best_recall:.4f}")
    print(f"F1 Score : {best_f1:.4f}")

    # 计算最终的平均定位误差
    final_avg_error = compute_average_error(all_test_predictions, all_test_labels, label_to_coord)
    print(f"Final Average Localization Error: {final_avg_error:.4f}m")
    sio.savemat(f'mlp_result/final_avg_error_{timestamp}.mat', {'final_avg_error': final_avg_error})
    # 保存最终预测结果
    sio.savemat(f'mlp_vis/locResult_{timestamp}.mat', {'loc_prediction': np.array(all_test_predictions)})
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_loc, label='Training loss', linewidth=2)
    plt.plot(test_loss_loc, label='Testing loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    #plt.title(f'训练和验证损失曲线（最佳验证准确率：{best_test_acc:.2f}%）', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # 保存高清图片
    loss_curve_path = f'mlp_vis/loss_curve_{timestamp}.png'
    plt.savefig(loss_curve_path, dpi=500, bbox_inches='tight')
    print(f'\n损失曲线已保存至：{loss_curve_path}')

    plt.show()

if __name__ == "__main__":
    main()