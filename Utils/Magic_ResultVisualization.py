import numpy as np
import json
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import math
import pandas as pd

class ResultGenerator():
    def __init__(self, path, test_flag, epoch, len_train_loader, len_vali_loader, len_test_loader=None, times_now=None):
        self.path = path
        self.test_flag = test_flag    # 表示是否进行测试的标志位
        self.epoch = epoch
        self.len_train_loader = len_train_loader
        self.len_vali_loader = len_vali_loader
        self.len_test_loader = len_test_loader
        self.times_now = times_now

    # ciallo: 绘制训练集、验证集的损失曲线，若存在测试集，也会绘制测试集损失曲线，并将其保存为图片
    def plot_train_validation_loss(self, train_loss_list, validation_loss_list, test_loss_list=None):
        fig, ax = plt.subplots()
        skip = 10
        x1 = np.array(range(1, len(train_loss_list) + 1), dtype=np.float32) / self.len_train_loader
        x2 = np.array(range(1, len(validation_loss_list) + 1), dtype=np.float32) / self.len_vali_loader
        plt.plot(x1[::skip], train_loss_list[::skip], color='#00BFFF', label="train loss")
        plt.plot(x2[::math.ceil(skip/2)], validation_loss_list[::math.ceil(skip/2)], color='#00FF7F', label="validation loss")
        if self.test_flag:
            x3 = np.array(range(1, len(test_loss_list) + 1), dtype=np.float32) / self.len_test_loader
            plt.plot(x3[::math.ceil(skip/2)], test_loss_list[::math.ceil(skip/2)], color='#EF8A43', label="test loss")
        plt.legend()
        plt.title(f"Loss")
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(math.ceil(self.epoch / 10)))
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(os.path.join(self.path, "{}_loss.png".format(self.times_now if self.times_now != None else 0)))
        plt.close()
    
    # ciallo: 类似于 plot_train_validation_loss，这个方法绘制训练集、验证集（以及测试集，如果存在）的准确率曲线，并保存为图片 
    def plot_train_validation_acc(self, train_acc_list, validation_acc_list, test_acc_list=None):
        x1 = np.array(range(1, len(train_acc_list) + 1), dtype=np.int16)
        x2 = np.array(range(1, len(validation_acc_list) + 1), dtype=np.int16)
        plt.plot(x1, train_acc_list, color='#00BFFF', label="train acc")
        plt.plot(x2, validation_acc_list, color='#00FF7F', label="validation acc")
        if self.test_flag:
            x3 = np.array(range(1, len(test_acc_list) + 1), dtype=np.int16)
            plt.plot(x3, test_acc_list, color='#EF8A43', label="test acc")
        plt.legend()
        plt.ylim((0, 1))
        plt.title("Acc")
        plt.xlabel("epoch")
        plt.ylabel("acc")
        plt.savefig(os.path.join(self.path, "{}_acc.png".format(self.times_now if self.times_now != None else 0)))
        plt.close()
    
    # ciallo: 绘制不同信噪比（SNR）下的模型准确率曲线，适用于通信相关任务 
    def plot_acc_of_dif_snr(self, validation_dict, test_dict=None):
        vali_list = []
        for snr in range(-20, 20, 2):
            vali_list.append(validation_dict["{}db".format(snr)][2])
        plt.figure(figsize=(12.8, 9.6), dpi=100)
        plt.plot(range(-20, 20, 2), vali_list, color='#00BFFF', marker='s', ms=10, linewidth=4,
                 label="validation accuracy")
        # for x, y in zip(range(-20, 20, 2), [sublist[index] for sublist in validation_acc_list]):
        #     plt.text(x, y, f'{y:0.2f}', fontsize=14, color='red', ha='center', va='bottom')
        if self.test_flag:
            test_list = []
            for snr in range(-20, 20, 2):
                test_list.append(test_dict["{}db".format(snr)][2])
            plt.plot(range(-20, 20, 2), test_list, color='#EF8A43', marker='*', ms=10, linewidth=4,
                 label="test accuracy")
            for x, y in zip(range(-20, 20, 2), test_list):
                plt.text(x, y, f'{y:0.2f}', fontsize=14, color='red', ha='center', va='bottom')
        plt.legend(fontsize="x-large", loc = "lower left")
        plt.title("SNR vs ACC")
        plt.xlabel("SNR(db)")
        plt.ylabel("accuracy")
        plt.grid(True, which='major', axis='both', color='gray', linestyle='dashed')
        plt.ylim((0, 1))
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
        plt.savefig(os.path.join(self.path, "{}_SNR_vs_ACC.png".format(self.times_now if self.times_now != None else 0)))
        plt.close()
    
    # ciallo: 将不同 SNR 下的准确率结果保存为 Excel 文件 
    def save_acc_of_dif_snr(self, dict):
        df = pd.DataFrame(dict)
        df.to_excel(os.path.join(self.path, "{}_SNR_vs_ACC.xlsx".format(self.times_now if self.times_now != None else 0)), index=False)

    # ciallo: 绘制混淆矩阵（Confusion Matrix）并可选择是否归一化 normalize：是否进行归一化显示。
    def __plot_confusion_matrix(self, confusion_matrix, classes=[], normalize=False, title='', cmap=plt.cm.Blues, save_filename=None):
        if classes == []:
            classes = [str(i) for i in range(len(confusion_matrix))]
        if normalize:
            confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)
            
        plt.figure(figsize=(6, 4))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title+" Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=-45)
        plt.yticks(tick_marks, classes)

        iters = np.reshape([[[i,j] for j in range(len(classes))] for i in range(len(classes))], (confusion_matrix.size,2))
        for i, j in iters:
            if normalize:
                plt.text(j, i, "{:.2f}".format(confusion_matrix[i, j]), va='center', ha='center')
            else:
                plt.text(j, i, format(confusion_matrix[i, j]), va='center', ha='center')

        plt.xlabel('Prediction')
        plt.ylabel('Real label')
        plt.tight_layout()
        plt.show()
        if save_filename is not None:
            plt.savefig(save_filename, dpi=600, bbox_inches = 'tight')
        plt.close()

    # ciallo: 调用 __plot_confusion_matrix 方法，绘制模型在测试集上的混淆矩阵并保存 
    def plot_confusion_matrix(self, confusion_matrix, classes_name=[]):
        self.__plot_confusion_matrix(confusion_matrix, classes=classes_name, normalize=False, title='', save_filename=os.path.join(self.path, "{}_matrix.png".format(self.times_now if self.times_now != None else 0)))
            
    