import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
# 设置 Matplotlib 的默认字体为一个支持中文的字体，例如 'SimHei'（黑体）
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class MLP(nn.Module):
    def __init__(self, input_features, layers_hidden, output_features=118):
        super(MLP, self).__init__()
        layers = []
        prev_features = input_features
        for hidden_features in layers_hidden:
            layers.append(nn.Linear(prev_features, hidden_features))
            layers.append(nn.ReLU())
            #layers.append(nn.Dropout(dropout_prob))
            prev_features = hidden_features
        layers.append(nn.Linear(prev_features, output_features))
        layers.append(nn.ReLU())
        #layers.append(nn.SiLU())
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
from sklearn.metrics import classification_report, precision_recall_fscore_support
### 导入新的 MLP 模型
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

### 全局常量变量
# ------------------------------------------------------------------------
INPUT_DIM = 520
OUTPUT_DIM = 118
VERBOSE = 1

# 解析参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-G", "--gpu_id", help="运行此脚本的 GPU 设备 ID；默认是 0；设置为负数表示使用 CPU（即不使用 GPU）", default=0, type=int)
    parser.add_argument("-R", "--random_seed", help="随机种子", default=0, type=int)
    parser.add_argument("-E", "--epochs", help="训练轮数；默认是 20", default=200, type=int)
    parser.add_argument("-B", "--batch_size", help="批处理大小；默认是 10", default=10, type=int)
    parser.add_argument("-T", "--training_ratio", help="训练数据占总体数据的比例：默认是 0.90", default=0.9, type=float)
    parser.add_argument("-N", "--neighbours", help="定位时考虑的（最近）邻居位置数；默认是 1", default=8, type=int)
    parser.add_argument("--scaling", help="用于包含邻居位置的阈值比例（即阈值=比例*最大值）；默认是 0.0", default=0.2, type=float)
    args = parser.parse_args()

    # 使用命令行参数设置变量
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    training_ratio = args.training_ratio
    #hidden_layers = [20,10]  # 调整隐藏层神经元数量
    hidden_layers = [512,256]  # 调整隐藏层神经元数量
    N = args.neighbours
    scaling = args.scaling

    # 初始化随机种子生成器
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(random_seed)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    # 加载并预处理数据
    train_df = pd.read_csv('../data/UJIIndoorLoc/trainingData2.csv', header=0)
    train_AP_features = scale(np.asarray(train_df.iloc[:, 0:INPUT_DIM]).astype(float), axis=1)
    train_df['REFPOINT'] = train_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])),
                                          axis=1)

    blds = np.unique(train_df[['BUILDINGID']])
    flrs = np.unique(train_df[['FLOOR']])
    for bld in blds:
        for flr in flrs:
            cond = (train_df['BUILDINGID'] == bld) & (train_df['FLOOR'] == flr)
            _, idx = np.unique(train_df.loc[cond, 'REFPOINT'], return_inverse=True)
            train_df.loc[cond, 'REFPOINT'] = idx

    blds = np.asarray(pd.get_dummies(train_df['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(train_df['FLOOR']))
    rfps = np.asarray(pd.get_dummies(train_df['REFPOINT']))
    train_labels = np.concatenate((blds, flrs, rfps), axis=1)

    train_val_split = np.random.rand(len(train_AP_features)) < training_ratio
    x_train = torch.tensor(train_AP_features[train_val_split], dtype=torch.float32).to(device)
    y_train = torch.tensor(train_labels[train_val_split], dtype=torch.float32).to(device)
    y_train = torch.clamp(y_train, 0, 1)  # 确保目标数据在 [0, 1] 范围内
    x_val = torch.tensor(train_AP_features[~train_val_split], dtype=torch.float32).to(device)
    y_val = torch.tensor(train_labels[~train_val_split], dtype=torch.float32).to(device)
    y_val = torch.clamp(y_val, 0, 1)  # 确保目标数据在 [0, 1] 范围内

    # 构建 MLP 模型
    model = MLP(input_features=INPUT_DIM, layers_hidden=hidden_layers).to(device)
    criterion = nn.BCELoss()
    weight_decay = 0.05    #最优
    #weight_decay = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=weight_decay) #最优
    #optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)
    # 现在optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=weight_decay)
    #weight_decay = 0.001  # 与kan一致
    #optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)  # 与kan一致

    #optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=weight_decay)   [20,10]
    #optimizer = optim.RMSprop(model.parameters(), lr=0.0001)
    #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.95)

    # 记录训练和验证损失
    train_losses = []
    val_losses = []
    start_time = timer()

    # 训练循环
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        # 增加正则化损失
        reg_loss = sum(torch.norm(param, 1) for param in model.parameters())
        total_loss = loss + 0.5 * reg_loss

        total_loss.backward()
        #loss.backward()
        optimizer.step()

        # 记录训练损失
        train_losses.append(loss.item())

        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
        scheduler.step(val_loss)

        val_losses.append(val_loss.item())
        #train_losses.append(total_loss.item())
        #val_losses.append(val_loss.item())
        print(f'第 {epoch + 1}/{epochs} 轮, 损失: {loss.item()}, 验证损失: {val_loss.item()}')

    end_time = timer()
    training_time = end_time - start_time
    print(f'训练时长: {training_time:.2f} 秒')
    # 保存模型
    now=datetime.datetime.now()
    torch.save(model.state_dict(), f'../results/pth/mlp_UJI_model_{now.strftime("%Y%m%d-%H%M%S")}.pth')

    # 绘制损失下降曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epochs')
    plt.ylabel('损失')
    plt.title('训练和验证损失下降曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png',dpi=500)
    plt.show()

    # 加载测试数据并评估
    test_df = pd.read_csv('../data/UJIIndoorLoc/validationData2.csv', header=0)
    test_AP_features = scale(np.asarray(test_df.iloc[:, 0:INPUT_DIM]).astype(float), axis=1)
    x_test_utm = np.asarray(test_df['LONGITUDE'])
    y_test_utm = np.asarray(test_df['LATITUDE'])
    blds = np.asarray(pd.get_dummies(test_df['BUILDINGID']))
    flrs = np.asarray(pd.get_dummies(test_df['FLOOR']))
    test_labels = np.concatenate((blds, flrs), axis=1)

    x_test = torch.tensor(test_AP_features, dtype=torch.float32).to(device)
    y_test = torch.tensor(test_labels, dtype=torch.float32).to(device)
    y_test = torch.clamp(y_test, 0, 1)  # 确保目标数据在 [0, 1] 范围内

    model.eval()
    with torch.no_grad():
        preds = model(x_test).cpu().numpy()

    blds_results = (np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
    acc_bld = blds_results.mean()
    flrs_results = (np.equal(np.argmax(test_labels[:, 3:8], axis=1), np.argmax(preds[:, 3:8], axis=1))).astype(int)
    acc_flr = flrs_results.mean()
    acc_bf = (blds_results * flrs_results).mean()
    print(f'acc_bld: {acc_bld}')
    print(f'acc_flr: {acc_flr}')
    print(f'acc_bf: {acc_bf}')
    # 计算建筑和楼层正确估计时的定位误差
    mask = np.logical_and(blds_results, flrs_results)
    x_test_utm = x_test_utm[mask]
    y_test_utm = y_test_utm[mask]
    rfps = (preds[mask])[:, 8:118]

    n_success = len(x_test_utm)  # 使用实际的成功数量
    blds = blds[mask]
    flrs = flrs[mask]

    n_loc_failure = 0
    sum_pos_err = 0.0
    sum_pos_err_weighted = 0.0
    idxs = np.argpartition(rfps, -min(N, rfps.shape[1]), axis=1)[:, -min(N, rfps.shape[1]):]  # 确保 N 不超过 rfps 的大小
    threshold = scaling * np.amax(rfps, axis=1)

    for i in range(n_success):
        xs = []
        ys = []
        ws = []
        for j in idxs[i]:
            if j >= rfps.shape[1]:
                continue
            rfp = np.zeros(110)
            rfp[j] = 1
            rows = np.where((train_labels == np.concatenate((blds[i], flrs[i], rfp))).all(axis=1))
            if rows[0].size > 0:
                if rfps[i][j] >= threshold[i]:
                    xs.append(train_df.iloc[rows[0][0], 520])  # LONGITUDE
                    ys.append(train_df.iloc[rows[0][0], 521])  # LATITUDE
                    ws.append(rfps[i][j])
        if len(xs) > 0:
            sum_pos_err += math.sqrt((np.mean(xs) - x_test_utm[i]) ** 2 + (np.mean(ys) - y_test_utm[i]) ** 2)
            sum_pos_err_weighted += math.sqrt(
                (np.average(xs, weights=ws) - x_test_utm[i]) ** 2 + (np.average(ys, weights=ws) - y_test_utm[i]) ** 2)
        else:
            n_loc_failure += 1
    mean_pos_err = sum_pos_err / (n_success - n_loc_failure)
    mean_pos_err_weighted = sum_pos_err_weighted / (n_success - n_loc_failure)
    loc_failure = n_loc_failure / n_success
    print(f'mean_pos_err: {mean_pos_err}')
    print(f'mean_pos_err_weighted: {mean_pos_err_weighted}')
    y_true_bld = np.argmax(test_labels[:, :3], axis=1)  # 建筑的真实标签
    y_pred_bld = np.argmax(preds[:, :3], axis=1)  # 建筑的预测标签
    y_true_flr = np.argmax(test_labels[:, 3:8], axis=1)  # 楼层的真实标签
    y_pred_flr = np.argmax(preds[:, 3:8], axis=1)  # 楼层的预测标签

    # 计算建筑分类的精确率、召回率、F1分数
    precision_bld, recall_bld, f1_bld, _ = precision_recall_fscore_support(y_true_bld, y_pred_bld, average='macro',zero_division=0)
    # 计算楼层分类的精确率、召回率、F1分数
    precision_flr, recall_flr, f1_flr, _ = precision_recall_fscore_support(y_true_flr, y_pred_flr, average='macro',zero_division=0)

    # 打印分类指标
    print("\n=== 建筑分类指标 ===")
    print(f"宏精确率 (Precision): {precision_bld:.4f}")
    print(f"宏召回率 (Recall): {recall_bld:.4f}")
    print(f"宏F1分数 (F1 Score): {f1_bld:.4f}")

    print("\n=== 楼层分类指标 ===")
    print(f"宏精确率 (Precision): {precision_flr:.4f}")
    print(f"宏召回率 (Recall): {recall_flr:.4f}")
    print(f"宏F1分数 (F1 Score): {f1_flr:.4f}")

    # 将结果写入文件
    now = datetime.datetime.now()
    path_out = f'../results/wbmlp{now.strftime("%Y%m%d-%H%M%S")}.org'
    with open(path_out, 'w') as f:
        f.write("#+STARTUP: showall\n")  # unfold everything when opening
        f.write("* 系统参数\n")
        f.write(f"  - Numpy 随机数种子: {random_seed}\n")
        f.write(f"  - 训练数据占总体数据的比例: {training_ratio}\n")
        f.write(f"  - 训练轮数: {epochs}\n")
        f.write(f"  - 批处理大小: {batch_size}\n")
        f.write(f"  - 邻居数: {N}\n")
        f.write(f"  - 阈值比例: {scaling}\n")
        f.write(f"  - 隐藏层结构: {hidden_layers}\n")
        f.write("* 性能\n")
        f.write(f"  - 建筑准确率: {acc_bld}\n")
        f.write(f"  - 楼层准确率: {acc_flr}\n")
        f.write(f"  - 建筑-楼层准确率: {acc_bf}\n")
        f.write(f"  - 定位失败率（给定正确的建筑/楼层）: {loc_failure}\n")
        f.write(f"  - 定位误差（米）: {mean_pos_err}\n")
        f.write(f"  - 加权定位误差（米）: {mean_pos_err_weighted}\n")
    path_out1 = f'../results/wbmlp_CM{now.strftime("%Y%m%d-%H%M%S")}.org'
    with open(path_out1, 'w') as f:
        f.write("=== 建筑分类指标 ===\n")
        f.write(f"  - 宏精确率 (Precision): {precision_bld:.4f}\n")
        f.write(f"  - 宏召回率 (Recall): {recall_bld:.4f}\n")
        f.write(f"  - 宏F1分数 (F1 Score): {f1_bld:.4f}\n")
        f.write("=== 楼层分类指标 ===\n")
        f.write(f"  - 宏精确率 (Precision): {precision_flr:.4f}\n")
        f.write(f"  - 宏召回率 (Recall): {recall_flr:.4f}\n")
        f.write(f"  - 宏F1分数 (F1 Score): {f1_flr:.4f}\n")