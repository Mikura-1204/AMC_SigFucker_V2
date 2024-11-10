#####------------------------------------------------------------------------------------#####
# ciallo illustration: 
# - This Magic Core is used to construct a Training Progress.  
#####------------------------------------------------------------------------------------#####  

import os
import json
import time
import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader

# ciallo: If train different kind of data with special DataLoader, change this to modify DataLoader Method.
from Data.Magic_Core_SigDataloader import Dataset_Sig

# ciallo: If you want to change Training Method and process, or LossFuc, change this.
from Utils.Magic_Normal_Tools import EarlyStopping, LearningRateAdjuster, save_model_structure_in_txt
from Utils.Magic_ResultVisualization import ResultGenerator
import Models
import Utils.Magic_LossFuction as myLF

# ciallo: Construct a Training Progress
class Exp():
    def __init__(self, args):
        self.args = args                                                # ciallo: get args from extern
        self.device = self.__acquire_device()                           # ciallo: choose device(GPU)
        self.path = None                                                # ciallo: model save path
        self.test_flag = True if args.data_split[2] != 0 else False     # ciallo: use Verify-Data
        self.class_names = None
        self.snrs = None
    
    # ciallo: Construct Model   
    def __build_model(self, pretrained=False, model_path=""):
        
        # ciallo GPT: 从Models中调用所需要的网络[字符串参数表示名字], 后面的分类数目被当作参数传入网络 比如-models.ResNet(10)
        model = getattr(Models, "{}".format(self.args.model))(self.args.n_class)
        
        
        if pretrained:
            if model_path != "":
                print("使用预训练模型喵 !", end='\n')
                model.load_state_dict(torch.load(model_path))
            else:
                raise TypeError

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model

    # ciallo: Loss Fuc construct (Three choose one)   
    def __build_criterion(self):
        if self.args.lossFunction == "CE":
            criterion = nn.CrossEntropyLoss()
        elif self.args.lossFunction == "GCE":
            criterion = myLF.GeneralizeCrossEntropy()
        elif self.args.lossFunction == "NCE":
            criterion = myLF.NoiseRobustCrossEntropy()
        return criterion

    # ciallo: Call GPU device
    def __acquire_device(self):
        if self.args.use_gpu and not self.args.use_multi_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda: {} 喵'.format(self.args.gpu),end='\n')
        elif self.args.use_gpu and self.args.use_multi_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{} 喵'.format(self.args.devices),end='\n')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        
        return device
    

    def __get_dataloader(self):
        args = self.args

        # ciallo: Pass in name and Split-Ratio
        dataset = Dataset_Sig(
            dataset_name=args.dataset,
            data_split=args.data_split,
        )
        
        train_dataset, val_dataset, test_dataset, self.class_names = dataset.get_dataset()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=False)
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False)
        
        print("训练集长度是 {} 喵, 验证集长度是 {} 喵".format(len(train_dataset), len(val_dataset)), end='')
        
        if self.test_flag:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False)
            print(", 测试集长度是 {} 喵".format(len(test_dataset)),end='\n')
        else:
            print("",end='\n')
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False)
        return train_loader, val_loader, test_loader

    # ciallo GPT: 执行验证过程，计算验证集的损失和准确率。
    def vali(self, loader, criterion):
        # ciallo GPT: 模型设置为评估模式
        self.model.eval()
        total_loss = []
        accuracy = {"correct":0, "total":0}
        with torch.no_grad():
            matrix = np.zeros(shape=(len(self.class_names), len(self.class_names)), dtype=np.int32)
            for i, (batch_x, batch_y) in enumerate(loader):
                # ciallo GPT: output是预测, label是正确的
                pred, true = self.__process_one_batch(batch_x, batch_y)
                # ciallo GPT: 找到每个样本的预测类别，即预测概率最大的类别
                batch_y_hat = torch.argmax(pred, 1)
                accuracy["correct"] += (batch_y_hat == true).sum().item()
                accuracy["total"] += batch_y.shape[0]
                loss = self.__cal_loss(pred, true, criterion)
                total_loss.append(loss.detach().item())
                for i in range(len(true)):
                    matrix[true[i]][batch_y_hat[i]] += 1
        # ciallo: 完成验证后，将模型切换回训练模式
        self.model.train()
        return total_loss, accuracy, matrix

    # ciallo: 训练过程：获取数据加载器，配置学习率调整、早停机制和结果生成器，逐步进行模型训练和验证。
    #         每个 epoch 中通过 __process_one_batch 处理数据，计算损失并更新模型参数。
    def train(self, setting, times_now):
        train_loader, vali_loader, test_loader = self.__get_dataloader()

        self.path = os.path.join(self.args.result_root_path, setting)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        # ciallo GPT: 设定并保存json文件
        with open(os.path.join(self.path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)

        train_steps = len(train_loader)
        
        # ciallo GPT: 使用三种辅助方法帮助训练 (包含可视化结果)
        #             LearningRateAdjuster 用于调整学习率
        #             EarlyStopping 用于在验证集性能不再提高时提前停止训练
        #             ResultGenerator 用于生成和保存训练结果
        lr_adjuster = LearningRateAdjuster(initial_lr=self.args.learning_rate, patience=self.args.patience, type=self.args.lradj)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0.001, data_parallel=self.args.use_multi_gpu, times_now=times_now)
        result_generator = ResultGenerator(self.path, self.test_flag, self.args.train_epochs, len(train_loader), len(vali_loader), len(test_loader), times_now)

        # ciallo GPT: 构建并保存模型结构
        self.model = self.__build_model().to(self.device)
        save_model_structure_in_txt(self.path, self.model)
        
        # ciallo GPT: 优化器与损失函数构建
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = self.__build_criterion()

        total_train_loss, total_vali_loss, total_test_loss = [], [], []
        total_train_acc, total_vali_acc, total_test_acc = [], [], []
        test_top_matrix = None
        
        # ciallo GPT: 单轮训练
        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            train_acc = {"correct":0, "total":0}
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1

                # ciallo GPT: 梯度清零 前向传播
                model_optim.zero_grad()
                pred, true = self.__process_one_batch(batch_x, batch_y)
                
                loss = self.__cal_loss(pred, true, criterion)
                train_loss.append(loss.item())
                batch_y_hat = torch.argmax(pred, 1)
                train_acc["correct"] += (batch_y_hat == true).sum().item()
                train_acc["total"] += batch_y.shape[0]

                # ciallo GPT: 每100个迭代打印当前的损失和估算剩余时间
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} 喵".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s 喵'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # ciallo GPT: 反向传播和优化
                loss.backward()
                model_optim.step()

            print("Epoch: {0} cost time: {1:.4f}s".format(epoch + 1, time.time() - epoch_time))
            vali_loss, vali_acc, _ = self.vali(vali_loader, criterion)
            total_train_loss.extend(train_loss)
            total_train_acc.append(train_acc["correct"]/train_acc["total"])
            total_vali_loss.extend(vali_loss)
            total_vali_acc.append(vali_acc["correct"]/vali_acc["total"])
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Vali Acc: {4:.4f}".format(
                epoch + 1, train_steps, np.average(train_loss), np.average(vali_loss), vali_acc["correct"]/vali_acc["total"]), end="")
            if self.test_flag:
                test_loss, test_acc, test_matrix = self.vali(test_loader, criterion)
                total_test_loss.extend(test_loss)
                total_test_acc.append(test_acc["correct"]/test_acc["total"])
                print(" Test Loss: {0:.7f} Test Acc: {1:.4f}".format(np.average(test_loss), test_acc["correct"]/test_acc["total"]))
            else:
                print("")
            best_model_path, save_flag = early_stopping(np.average(vali_loss), vali_acc["correct"]/vali_acc["total"], self.model, self.path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if save_flag:
                test_top_matrix = test_matrix
            lr_adjuster.rate_decay_with_patience(model_optim, early_stopping.counter)
        
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, best_model_path)
        
        # self.save_reasult()
        result_generator.plot_train_validation_loss(total_train_loss, total_vali_loss, total_test_loss)
        result_generator.plot_train_validation_acc(total_train_acc, total_vali_acc, total_test_acc)
        result_generator.plot_confusion_matrix(test_top_matrix, self.class_names)

        return self.model
    
    def __process_one_batch(self, batch_x, batch_y, inverse = False):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        outputs = self.model(batch_x)

        return outputs, batch_y
    
    def __cal_loss(self, pred, true, criterion, auxiliary = None):
        if self.args.lossFunction == "CE":
            loss = criterion(pred, true)
        elif self.args.lossFunction == "GCE":
            loss = criterion(pred, true, auxiliary)
        elif self.args.lossFunction == "NCE":
            loss = criterion(pred, true, auxiliary)

        return loss
    
    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def save_reasult(self):
        pass
