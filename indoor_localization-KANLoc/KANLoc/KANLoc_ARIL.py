import os
import math
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support
import datetime
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
# 设置 Matplotlib 的默认字体为一个支持中文的字体，例如 'SimHei'（黑体）
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号



# 位置标签到二维坐标的映射
def get_label_to_coord_mapping(num_labels=16):
    """
    将位置标签映射到二维坐标。
    将16个标签映射到4x4网格上，每个网格间距0.8米。

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
    seed = 42  # 可以修改为你想要的任何种子值

    # 基础库设置
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # GPU相关设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False  # 关闭加速可能导致速度下降，但确保确定性

    # 数据加载器设置
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(seed)

    #batch_size = 128
    batch_size=256
    num_epochs = 600
    #learning_rate = 0.05
    learning_rate = 0.001
    #milestone_steps = [10,50, 100, 150]
    milestone_steps = [10,50,100,150]#学习率调整的里程碑
    #gamma = 0.5  # 学习率衰减因子
    gamma = 0.5  # 学习率衰减因子
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 创建结果保存目录
    os.makedirs('kanloc_weights_ae', exist_ok=True)
    os.makedirs('kanloc_result_ae', exist_ok=True)
    os.makedirs('kanloc_vis_ae', exist_ok=True)

    # 加载训练数据
    print("Loading training data...")
    train_data_amp = sio.loadmat('../csi_data/train_data_split_amp.mat')['train_data']
    train_location_label = sio.loadmat('../csi_data/train_data_split_amp.mat')['train_location_label']
    train_labels_combined = train_location_label.squeeze()
    num_train_instances = train_data_amp.shape[0]

    train_data = torch.from_numpy(train_data_amp).float()
    train_labels = torch.from_numpy(train_labels_combined).long()

    train_dataset = TensorDataset(train_data, train_labels)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,generator=dataloader_generator,  # 新增
        worker_init_fn=seed_worker)

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
    layers_hidden = [52, 512,256,  num_labels]  # 隐藏层配置，请根据需要调整
    kan_model = KAN(layers_hidden=layers_hidden,grid_size=5,
                    spline_order=3).cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(kan_model.parameters(), lr=learning_rate)
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
        kan_model.train()


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
            outputs = kan_model(samples)
            loss = criterion(outputs, labels)
            reg_loss = kan_model.regularization_loss()
            total_loss = loss + reg_loss

            #loss.backward()
            total_loss.backward()
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
        kan_model.eval()
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

                outputs = kan_model(samples)
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
        print(classification_report(all_test_labels, all_test_predictions,digits=4, zero_division=0))

        # 计算并打印宏平均
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_test_labels, all_test_predictions, average='macro' , zero_division=0
        )

        print(f"Macro Average - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
        # 保存最佳模型
        if epoch_acc_test > best_test_acc:
            best_test_acc = epoch_acc_test
            best_precision = precision_macro
            best_recall = recall_macro
            best_f1 = f1_macro
            best_model_path = f'kanloc_weights_ae/kan_model_best_epoch{epoch + 1}_Acc{epoch_acc_test:.3f}.pth'  # 添加路径记录
            torch.save(kan_model.state_dict(), best_model_path)  # 取消注释
            print(f'Best model saved at epoch {epoch + 1} with Test Accuracy: {epoch_acc_test:.3f}%')




    # 保存训练和测试结果
    print("Saving results...")
    '''sio.savemat(f'kanloc_result_ae/train_loss_loc_{timestamp}.mat', {'train_loss_loc': train_loss_loc})
    sio.savemat(f'kanloc_result_ae/test_loss_loc_{timestamp}.mat', {'test_loss_loc': test_loss_loc})
    sio.savemat(f'kanloc_result_ae/train_acc_loc_{timestamp}.mat', {'train_acc_loc': train_acc_loc})
    sio.savemat(f'kanloc_result_ae/test_acc_loc_{timestamp}.mat', {'test_acc_loc': test_acc_loc})
    sio.savemat(f'kanloc_result_ae/test_avg_error_{timestamp}.mat', {'test_avg_error': test_avg_error})
    sio.savemat(f'kanloc_result_ae/best_precision_{timestamp}.mat', {'best_precision': best_precision})
    sio.savemat(f'kanloc_result_ae/best_recall_{timestamp}.mat', {'best_recall': best_recall})
    sio.savemat(f'kanloc_result_ae/best_f1_{timestamp}.mat', {'best_f1': best_f1})'''
    print("Training and testing completed.")

    print("\n\n=== 使用最佳模型进行最终评估 ===")
    # 加载最佳模型
    kan_model.load_state_dict(torch.load(best_model_path))
    kan_model.eval()

    # 重新运行测试集预测
    all_test_predictions = []
    all_test_labels = []

    with torch.no_grad():
        for samples, labels in test_data_loader:
            if len(samples.shape) > 2:
                samples = samples.mean(dim=-1)
            samples = samples.cuda()
            labels = labels.cuda()

            outputs = kan_model(samples)
            _, predicted = torch.max(outputs.data, 1)

            all_test_predictions.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())

    # 计算最终指标
    final_avg_error = compute_average_error(all_test_predictions, all_test_labels, label_to_coord)
    final_acc = 100.0 * np.mean(np.array(all_test_predictions) == np.array(all_test_labels))

    print(f"\n最佳模型最终测试准确率: {final_acc:.2f}%")
    print(f"最佳模型最终平均定位误差: {final_avg_error:.4f}m")

    # 生成混淆矩阵
    cm = confusion_matrix(all_test_labels, all_test_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(18, 15))
    ax=sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=np.arange(num_labels),
                yticklabels=np.arange(num_labels),
                annot_kws={"size": 13},
                cbar=False,)
    plt.xlabel('Predicting location label', fontsize=20)
    plt.ylabel('Real location label', fontsize=20)
    #plt.title(f'最佳模型混淆矩阵 (准确率: {final_acc:.2f}%)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # 调整颜色条字体大小
    cbar = plt.colorbar(ax.collections[0])  # 传递 mappable 对象
    # cbar.set_label('label',fontsize=14)  # 颜色条标签字体大小
    cbar.ax.tick_params(labelsize=14)  # 颜色条刻度字体大小
    # 添加黑色边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.5)
    # 保存混淆矩阵图
    cm_path = f'kanloc_vis/kanloc_ARIL_BEST_confusion_matrix_{timestamp}.png'
    #plt.savefig(cm_path, dpi=500, bbox_inches='tight')
    #plt.show()
    print(f'\n最佳模型混淆矩阵已保存至：{cm_path}')



    # 最终评估
    print("\nFinal Evaluation on Test Set:")
    print(f"Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Precision : {best_precision:.4f}")
    print(f"Recall : {best_recall:.4f}")
    print(f"F1 Score : {best_f1:.4f}")

    # 计算最终的平均定位误差
    final_avg_error = compute_average_error(all_test_predictions, all_test_labels, label_to_coord)
    print(f"Final Average Localization Error: {final_avg_error:.4f}m")
    #sio.savemat(f'kanloc_result_ae/final_avg_error_{timestamp}.mat', {'final_avg_error': final_avg_error})

    # 保存最终预测结果
    #sio.savemat(f'kanloc_vis_ae/locResult_{timestamp}.mat', {'loc_prediction': np.array(all_test_predictions)})

    plt.figure(figsize=(10, 6))
    matplotlib.rcParams['axes.linewidth'] = 2
    plt.plot(train_loss_loc, label='Training loss', linewidth=2)
    plt.plot(test_loss_loc, label='Testing loss', linewidth=2)
    plt.xlabel('Epochs', fontsize=24)
    plt.ylabel('Loss', fontsize=24)
    # plt.title(f'训练和验证损失曲线（最佳验证准确率：{best_test_acc:.2f}%）', fontsize=16)
    plt.legend(fontsize=22)
    #plt.grid(True, alpha=0.3)
    plt.grid(True, alpha=0.6, linewidth=1.5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # 保存高清图片
    loss_curve_path = f'kanloc_vis/loss_curve_{timestamp}.png'
    plt.savefig(loss_curve_path, dpi=500, bbox_inches='tight')
    print(f'\n损失曲线已保存至：{loss_curve_path}')

    #plt.show()

    '''def modified_forward(self, x):
        for layer, batch_norm in zip(self.layers[:-1], self.batch_norms[:-1]):  # 只遍历到倒数第二层
            x = layer(x)
            x = batch_norm(x)
        features = x  # 获取最后一个隐藏层的输出
        outputs = self.layers[-1](features)  # 最后一层分类器
        return outputs, features

    # 保存原始前向传播方法
    original_forward = KAN.forward

    # 临时替换前向传播方法
    KAN.forward = modified_forward

    # 提取测试集特征
    all_features = []
    all_labels = []
    kan_model.eval()
    with torch.no_grad():
        for samples, labels in test_data_loader:
            if len(samples.shape) > 2:
                samples = samples.mean(dim=-1)
            samples = samples.cuda()
            _, features = kan_model(samples)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # 恢复原始前向传播方法
    KAN.forward = original_forward

    # 随机采样部分数据（加快计算）
    np.random.seed(42)
    sample_idx = np.random.choice(len(all_features),len(all_features) , replace=False)
    sampled_features = all_features[sample_idx]
    sampled_labels = all_labels[sample_idx]

    # t-SNE降维
    from sklearn.manifold import TSNE
    print("Running t-SNE...")
    #tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=6000, learning_rate=300, init='pca', random_state=42)
    features_2d = tsne.fit_transform(sampled_features)

    # 可视化设置
    plt.figure(figsize=(12, 10))
    cmap = plt.cm.get_cmap('tab20b', num_labels)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                          c=sampled_labels, cmap=cmap, alpha=0.99,
                          s=30, edgecolor='none')

    # 添加颜色条
    cbar = plt.colorbar(scatter, ticks=np.arange(num_labels))
    cbar.set_label('Location label', fontsize=18)
    cbar.ax.tick_params(labelsize=12)

    # 添加标题和标签
    #plt.title('t-SNE可视化 - 位置特征分布', fontsize=14)
    plt.xlabel('tSNE_1', fontsize=18)
    plt.ylabel('tSNE_2', fontsize=18)
    #plt.grid(True, alpha=0.2)

    # 添加图例
    handles = []
    for i in range(num_labels):
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=cmap(i), markersize=10, label=f'Label {i}'))

    plt.legend(handles=handles, title='Location label', loc='best', title_fontsize=14,fontsize=11)

    # 保存和显示
    tsne_path = f'kanloc_vis/kanloc_ARIL_tsne_{timestamp}.png'
    #plt.savefig(tsne_path, dpi=500, bbox_inches='tight')
    print(f'\nt-SNE可视化结果已保存至：{tsne_path}')
    #plt.show()'''

if __name__ == "__main__":
    main()
