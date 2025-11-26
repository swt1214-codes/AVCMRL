from __future__ import print_function
import csv
import math
import os

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim
import torch.nn.init as init
import sklearn.metrics as metrics
import numpy as np
from read_data_val import RBPDATASET
import config as config
from torch.utils.data import DataLoader


pic_path = 'D:/DeskTop/MVRL_codes_swt/pics'
models_path = 'D:/DeskTop/MVRL_codes_swt/model_path'


class SharedAndSpecificLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(SharedAndSpecificLoss, self).__init__()
        self.smoothing = smoothing

    @staticmethod
    def orthogonal_loss(shared, specific):
        # 强制不同特征空间之间的正交性，以促进特征分离,减少冗余信息
        shared = shared - shared.mean()
        specific = specific - specific.mean()
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        correlation_matrix = shared.t().matmul(specific)
        cost = torch.sum(correlation_matrix ** 2) / shared.size(1)
        return cost

    @staticmethod
    def similarity_loss(shared_1, shared_2, shared_3):
        # 最大化不同视图共享特征之间的相似性
        sim_12 = F.cosine_similarity(shared_1, shared_2, dim=1).mean()
        sim_13 = F.cosine_similarity(shared_1, shared_3, dim=1).mean()
        sim_23 = F.cosine_similarity(shared_2, shared_3, dim=1).mean()
        loss = (1 - sim_12) + (1 - sim_13) + (1 - sim_23)
        return loss


    def test_loss(self, classification_output, target):
        return F.cross_entropy(classification_output, target)

    def forward(self, epoch, total_epoch, level_output, classification_output, target):

        # 正交损失
        orthogonal_loss = 0.0
        for i in range(len(level_output)):
            view1_loss = self.orthogonal_loss(level_output[i][0], level_output[i][3])
            view2_loss = self.orthogonal_loss(level_output[i][1], level_output[i][4])
            view3_loss = self.orthogonal_loss(level_output[i][2], level_output[i][5])
            orthogonal_loss += view1_loss + view2_loss + view3_loss

        # 相似性损失
        similarity_loss = 0.0
        for i in range(len(level_output)):
            sim_loss = self.similarity_loss(
                level_output[i][3], level_output[i][4], level_output[i][5]
            )
            similarity_loss += sim_loss

        # 分类损失
        classification_loss = F.cross_entropy(classification_output, target, label_smoothing=self.smoothing)

        alpha = max(0.2, 0.5 * (1 + math.cos(math.pi * epoch / total_epoch)))
        beta = 1 - alpha


        total_loss = (
                alpha * orthogonal_loss +  # 0.2
                beta * similarity_loss +  # 0.5
                1.0 * classification_loss
        )
        return total_loss


class AttrProxy(object):
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, index):
        return getattr(self.module, self.prefix + str(index))


class ViewProcessor(nn.Module):
    def __init__(self, input_dim, n_units, feature_size):
        super().__init__()

        self.shared_path = nn.Sequential(
            nn.Linear(input_dim, n_units[0]),
            nn.ReLU(),
            nn.Linear(n_units[0], n_units[1]),
            nn.ReLU(),
            nn.Linear(n_units[1], feature_size),
            nn.ReLU()
        )


        self.specific_path = nn.Sequential(
            nn.Linear(input_dim, n_units[0]),
            nn.ReLU(),
            nn.Linear(n_units[0], n_units[1]),
            nn.ReLU(),
            nn.Linear(n_units[1], feature_size),
            nn.ReLU()
        )

    def init_params(self):
        for layer in self.shared_path:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)
        for layer in self.specific_path:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)

    def forward(self, x):
        shared = self.shared_path(x)
        specific = self.specific_path(x)
        return shared, specific


class CrossViewAttention(nn.Module):

    def __init__(self, current_dim, feature_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_size // num_heads


        self.query = nn.Linear(current_dim, feature_size)


        self.key_view2 = nn.Linear(feature_size, feature_size)
        self.key_view3 = nn.Linear(feature_size, feature_size)
        self.value_view2 = nn.Linear(feature_size, feature_size)
        self.value_view3 = nn.Linear(feature_size, feature_size)


        self.out_proj = nn.Linear(feature_size, current_dim)
        self.norm = nn.LayerNorm(current_dim)

    def forward(self, current_view, view2_spec, view3_spec):

        B, _ = current_view.size()

        # 生成Q [B, H, D]
        q = self.query(current_view).view(B, self.num_heads, self.head_dim)

        # 分视图投影 [B, H, D]
        k2 = self.key_view2(view2_spec).view(B, self.num_heads, self.head_dim)
        k3 = self.key_view3(view3_spec).view(B, self.num_heads, self.head_dim)
        v2 = self.value_view2(view2_spec).view(B, self.num_heads, self.head_dim)
        v3 = self.value_view3(view3_spec).view(B, self.num_heads, self.head_dim)


        keys = torch.stack([k2, k3], dim=2)
        values = torch.stack([v2, v3], dim=2)


        q = q.unsqueeze(2)  # [B, H, 1, D]
        keys = keys.transpose(-1, -2)  # [B, H, D, 2]


        attn_scores = (q @ keys) / (self.head_dim ** 0.5)  # [B, H, 1, 2]
        attn_weights = F.softmax(attn_scores, dim=-1)


        attn_output = attn_weights @ values  # [B, H, 1, D]
        attn_output = attn_output.squeeze(2).reshape(B, -1)  # [B, H*D]


        output = self.norm(current_view + self.out_proj(attn_output))
        return output, attn_weights


# attention fusion communication block
class ResCommunicationBlock(nn.Module):
    def __init__(self, view_size=[64, 64, 64], n_units=[128, 64], feature_size=64, num_heads=4):
        super().__init__()

        shared_params = {
            'n_units': n_units,
            'feature_size': feature_size
        }

        self.view1_processor = ViewProcessor(view_size[0], **shared_params)
        self.view2_processor = ViewProcessor(view_size[1], **shared_params)
        self.view3_processor = ViewProcessor(view_size[2], **shared_params)


        self.fusion1 = CrossViewAttention(
            current_dim=view_size[0],
            feature_size=feature_size,
            num_heads=num_heads
        )
        self.fusion2 = CrossViewAttention(
            current_dim=view_size[1],
            feature_size=feature_size,
            num_heads=num_heads
        )
        self.fusion3 = CrossViewAttention(
            current_dim=view_size[2],
            feature_size=feature_size,
            num_heads=num_heads
        )

        # 归一化
        self.norm = nn.LayerNorm(feature_size)

    def init_params(self):
        # 初始化各视图参数
        self.view1_processor.init_params()
        self.view2_processor.init_params()
        self.view3_processor.init_params()

        # 初始化注意力模块
        for m in [self.fusion1, self.fusion2, self.fusion3]:
            # 初始化各视图独立的Key/Value投影层
            init.kaiming_normal_(m.key_view2.weight)
            init.kaiming_normal_(m.key_view3.weight)
            init.kaiming_normal_(m.value_view2.weight)
            init.kaiming_normal_(m.value_view3.weight)

            # 初始化其他参数
            init.kaiming_normal_(m.query.weight)
            init.kaiming_normal_(m.out_proj.weight)

    def forward(self, v1, v2, v3):

        v1_shared, v1_spec = self.view1_processor(v1)
        v2_shared, v2_spec = self.view2_processor(v2)
        v3_shared, v3_spec = self.view3_processor(v3)


        fused1, attn1 = self.fusion1(v1, v2_spec, v3_spec)
        fused2, attn2 = self.fusion2(v2, v1_spec, v3_spec)
        fused3, attn3 = self.fusion3(v3, v1_spec, v2_spec)


        v1_new = self.norm(v1 + fused1)
        v2_new = self.norm(v2 + fused2)
        v3_new = self.norm(v3 + fused3)


        attn_tensor = torch.stack([attn1.mean(dim=1),
                                   attn2.mean(dim=1),
                                   attn3.mean(dim=1)], dim=1)  # shape: (B,3,2)

        return v1_new, v2_new, v3_new, v1_spec, v2_spec, v3_spec, v1_shared, v2_shared, v3_shared, attn_tensor


# whole module
class MultipleRoundsCommunication(nn.Module):

    def __init__(self, level_num=5, original_view_size=[252,35,400], view_size=[128, 128, 128], feature_size=128,
                 n_units=[128, 64], c_n_units=[64, 32], class_num=2, similarity_threshold = 0.95):

        super(MultipleRoundsCommunication, self).__init__()


        shared_params = {
            'n_units': n_units,
            'feature_size': 128
        }
        self.similarity_threshold = similarity_threshold
        self.actual_levels = level_num


        self.view1_processor = ViewProcessor(view_size[0], **shared_params)
        self.view2_processor = ViewProcessor(view_size[1], **shared_params)
        self.view3_processor = ViewProcessor(view_size[2], **shared_params)

        # View1 Input
        self.input1_l1 = nn.Linear(original_view_size[0], n_units[0])
        self.input1_l2 = nn.Linear(n_units[0], n_units[1])
        self.input1_l3 = nn.Linear(n_units[1], view_size[0])

        # View2 Input
        self.input2_l1 = nn.Linear(original_view_size[1], n_units[0])
        self.input2_l2 = nn.Linear(n_units[0], n_units[1])
        self.input2_l3 = nn.Linear(n_units[1], view_size[1])

        # View3 Input
        self.input3_l1 = nn.Linear(original_view_size[2], n_units[0])
        self.input3_l2 = nn.Linear(n_units[0], n_units[1])
        self.input3_l3 = nn.Linear(n_units[1], view_size[2])

        # Communication block
        self.level_num = level_num
        for i_th in range(self.level_num):
            basic_model = ResCommunicationBlock(view_size=[view_size[0], view_size[1], view_size[2]], n_units=n_units,
                                                feature_size=feature_size)
            self.add_module('level_' + str(i_th), basic_model)
        self.levels = AttrProxy(self, 'level_')

        self.input_dim = sum(view_size) + feature_size

        self.classification_l1 = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.classification_l2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.classification_l3 = nn.Linear(128, class_num)

    def init_params(self):
        # Input init
        init.kaiming_normal_(self.input1_l1.weight)
        init.kaiming_normal_(self.input1_l2.weight)
        init.kaiming_normal_(self.input1_l3.weight)

        init.kaiming_normal_(self.input2_l1.weight)
        init.kaiming_normal_(self.input2_l2.weight)
        init.kaiming_normal_(self.input2_l3.weight)

        init.kaiming_normal_(self.input3_l1.weight)
        init.kaiming_normal_(self.input3_l2.weight)
        init.kaiming_normal_(self.input3_l3.weight)

        # init module
        for i_th in range(self.level_num):
            name = 'level_' + str(i_th)
            level = self._modules[name]
            level.init_params()

        # Classification init
        init.kaiming_normal_(self.classification_l1.weight)
        init.kaiming_normal_(self.classification_l2.weight)
        init.kaiming_normal_(self.classification_l3.weight)

    def encode(self, original_view1_input, original_view2_input, original_view3_input):
        # Input View1
        input1_output = F.relu(self.input1_l1(original_view1_input))
        input1_output = F.relu(self.input1_l2(input1_output))
        input1 = F.relu(self.input1_l3(input1_output))

        # Input View2
        input2_output = F.relu(self.input2_l1(original_view2_input))
        input2_output = F.relu(self.input2_l2(input2_output))
        input2 = F.relu(self.input2_l3(input2_output))

        # Input View3
        input3_output = F.relu(self.input3_l1(original_view3_input))
        input3_output = F.relu(self.input3_l2(input3_output))
        input3 = F.relu(self.input3_l3(input3_output))


        view1_updates = []
        view2_updates = []
        view3_updates = []


        level_output = []


        level_attentions = []


        prev_v1, prev_v2, prev_v3 = input1, input2, input3


        v1_shared, v1_spec = self.view1_processor(input1)
        v2_shared, v2_spec = self.view2_processor(input2)
        v3_shared, v3_spec = self.view3_processor(input3)


        final_v1 = v1_spec.clone().detach()
        final_v2 = v2_spec.clone().detach()
        final_v3 = v3_spec.clone().detach()

        actual_levels = 0


        for i_th in range(self.level_num):
            level = self._modules[f'level_{i_th}']
            input1_new, input2_new, input3_new, vs1, vs2, vs3, vsh1, vsh2, vsh3, attn = level(input1, input2, input3)

            level_attentions.append(attn)


            view1_updates.append(input1_new)
            view2_updates.append(input2_new)
            view3_updates.append(input3_new)
            actual_levels += 1


            prev_info = [prev_v1, prev_v2, prev_v3]  # 上一轮的视图特征
            curr_info = [input1_new, input2_new, input3_new]  # 当前更新轮后的视图特征


            view_sim = self.compute_similarity(prev_info, curr_info)

            if view_sim >= self.similarity_threshold and i_th >= 1:
                print(f'通信停止，当前视图间相似度为：{view_sim}， 通信交流轮次为：{actual_levels}！')
                break


            prev_v1, prev_v2, prev_v3 = input1_new, input2_new, input3_new
            input1, input2, input3 = input1_new, input2_new, input3_new


            level_output.append([vs1, vs2, vs3, vsh1, vsh2, vsh3])


        level_attn_tensor = torch.stack(level_attentions, dim=1)


        for update in view1_updates:
            final_v1 += update
        for update in view2_updates:
            final_v2 += update
        for update in view3_updates:
            final_v3 += update

        return level_output, final_v1, final_v2, final_v3, [vsh1, vsh2, vsh3], level_attn_tensor

    def forward(self, original_view1_input, original_view2_input, original_view3_input):
        level_output, final_v1, final_v2, final_v3, last_share, attn_weights = self.encode(original_view1_input, original_view2_input,
                                                                             original_view3_input)

        all_share_mean = (last_share[0] + last_share[1] + last_share[2]) / 3


        classification_input = torch.cat([
            final_v1,
            final_v2,
            final_v3,
            all_share_mean
        ], dim=1)


        classification_output = F.relu(self.bn1(self.classification_l1(classification_input)))
        classification_output = self.dropout(classification_output)
        classification_output = F.relu(self.bn2(self.classification_l2(classification_output)))
        classification_output = self.dropout(classification_output)
        classification_output = self.classification_l3(classification_output)

        return level_output, classification_output, attn_weights


    def compute_similarity(self, prev_shared, curr_shared):

        prev_shared = torch.stack(prev_shared) if isinstance(prev_shared, list) else prev_shared
        curr_shared = torch.stack(curr_shared) if isinstance(curr_shared, list) else curr_shared

        shared_similarity = F.cosine_similarity(prev_shared, curr_shared, dim=1).mean()

        return shared_similarity


def main():
    # Hyper Parameters
    EPOCH = config.MAX_EPOCH
    BATCH_SIZE = config.BATCH_SIZE
    USE_GPU = config.USE_GPU
    LEVEL_NUM = 5


    train_data = RBPDATASET(set_name='train')
    test_data = RBPDATASET(set_name='test')

    print("================train_data======================")
    print("view1:", len(train_data.view1_data), train_data.view1_data.shape)
    print("view2:", len(train_data.view2_data), train_data.view2_data.shape)
    print("view3:", len(train_data.view3_data), train_data.view3_data.shape)
    print("label:", len(train_data.labels), train_data.labels.shape)
    print("================test_data=================")
    print("view1:", len(test_data.view1_data), test_data.view1_data.shape)
    print("view2:", len(test_data.view2_data), test_data.view2_data.shape)
    print("view3:", len(test_data.view3_data), test_data.view3_data.shape)
    print("label:", len(test_data.labels), test_data.labels.shape)

    # Build Model
    model = MultipleRoundsCommunication(level_num=LEVEL_NUM,
                                        original_view_size=[252,35,400],
                                        view_size=[128, 128, 128], feature_size=128, n_units=[128, 128],
                                        c_n_units=[128, 64],
                                        class_num=2, similarity_threshold= 0.95)
    model.init_params()
    #print(model)
    #model.show()

    if USE_GPU:
        model = model.cuda()

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98),
                                  weight_decay=0.01)  # lr = 0.0003,wd=0.0001
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=200,
        T_mult=2,
        eta_min=1e-6
    )
    loss_function = SharedAndSpecificLoss()

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    print("Training...")

    # 保存数据，创建文件，写入存储的字段
    csv_path = "../results/RBPpred_draw.csv"
    csv_columns = [
        'Epoch', 'Total_Epochs', 'Training_Loss', 'Training_Acc',
        'Testing_Loss', 'test_Acc', 'test_F1'
    ]

    if not os.path.isfile(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_columns)

    train_loss_ = []
    train_acc_ = []

    test_loss_ = []
    test_acc_ = []
    test_f1_ = []



    for epoch in range(EPOCH+1):
        train_total_acc = 0.0
        train_total_loss = 0.0
        train_total = 0.0
        model.train()

        # training=====================================================
        for train_iteration_index, train_data in enumerate(train_loader):
            page_input, link_input, view3_input, train_labels = train_data
            train_labels = torch.squeeze(train_labels)

            if USE_GPU:
                page_input = Variable(page_input.cuda())
                link_input = Variable(link_input.cuda())
                view3_input = Variable(view3_input.cuda())
                train_labels = train_labels.type(torch.LongTensor).cuda()
            else:
                page_input = Variable(page_input).type(torch.FloatTensor)
                link_input = Variable(link_input).type(torch.FloatTensor)
                view3_input = Variable(view3_input).type(torch.FloatTensor)
                train_labels = Variable(train_labels).type(torch.LongTensor)


            optimizer.zero_grad()
            level_output, classification_output, attn_weights = model(page_input, link_input, view3_input)

            loss = loss_function(epoch, EPOCH, level_output=level_output, classification_output=classification_output,
                                 target=train_labels)
            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(classification_output.data, 1)
            train_total_acc += (predicted == train_labels.data).sum()
            train_total += len(train_labels)
            train_total_loss += loss.item()

        train_loss_.append(train_total_loss / train_total)

        if config.USE_GPU:
            train_acc_.append(train_total_acc.cpu().numpy() / train_total)
        else:
            train_acc_.append(train_total_acc.numpy() / train_total)


        scheduler.step()

        # testing=========================================================
        model.eval()

        test_total_loss = 0.0
        test_predict_labels = []
        test_ground_truth = []

        with torch.no_grad():
            for iteration_index, test_data in enumerate(test_loader):
                test_page_inputs, test_link_inputs, view3_input, test_labels = test_data
                test_labels = torch.squeeze(test_labels)


                if USE_GPU:
                    test_page_inputs = Variable(test_page_inputs.cuda())
                    test_link_inputs = Variable(test_link_inputs.cuda())
                    view3_input = Variable(view3_input.cuda())
                    test_labels = test_labels.cuda()

                    level_output, classification_output, attn_weights = \
                        model(test_page_inputs, test_link_inputs, view3_input)


                    _, predicted = torch.max(classification_output.data, 1)
                    test_predict_labels.extend(list(predicted.cpu().numpy()))
                    test_ground_truth.extend(list(test_labels.data.cpu().numpy()))
                else:
                    test_page_inputs = Variable(test_page_inputs).type(torch.FloatTensor)
                    test_link_inputs = Variable(test_link_inputs).type(torch.FloatTensor)
                    test_labels = Variable(test_labels).type(torch.LongTensor)
                    view3_input = Variable(view3_input).type(torch.FloatTensor)

                    level_output, classification_output, attn_weights = \
                        model(test_page_inputs, test_link_inputs, view3_input)


                    _, predicted = torch.max(classification_output.data, 1)
                    test_predict_labels.extend(list(predicted.numpy()))
                    test_ground_truth.extend(list(test_labels.data.numpy()))


                    loss = loss_function.test_loss(classification_output, test_labels)
                    test_total_loss += loss.item() * test_labels.size(0)


            test_ground_truth = np.array(test_ground_truth).astype(int)
            test_predict_labels = np.array(test_predict_labels).astype(int)


            test_loss = test_total_loss / len(test_loader.dataset)
            test_loss_.append(test_loss)

        # calculate acc and f1
        a_acc = metrics.accuracy_score(test_ground_truth, test_predict_labels)
        test_acc_.append(a_acc)
        result = metrics.classification_report(test_ground_truth, test_predict_labels, digits=5, output_dict=True,
                                               zero_division=0)
        a_f1 = result['weighted avg']['f1-score']
        test_f1_.append(a_f1)


        print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.5f, '
              'Testing Loss : %.3f, '
              'test Acc: %.5f, test F1: %.5f'

              % (epoch + 1, EPOCH, train_loss_[epoch], train_acc_[epoch],
                 test_loss_[epoch], test_acc_[epoch], test_f1_[epoch]
                 )
              )

        row_data = [
            epoch + 1,  # Epoch
            EPOCH,  # Total_Epochs
            float(train_loss_[epoch]),  # Training_Loss
            float(train_acc_[epoch]),  # Training_Acc
            float(test_loss_[epoch]), # Test_Loss
            float(test_acc_[epoch]), # Test_Acc
            float(test_f1_[epoch])  # Test_F1
        ]
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)


    final_acc = np.mean(np.sort(test_acc_)[-2:])
    final_f1 = np.mean(np.sort(test_f1_)[-2:])
    return final_acc, final_f1


if __name__ == "__main__":
    acc_list = []
    f1_list = []
    for i in range(config.ITERATIONS):
        acc, f1 = main()
        acc_list.append(acc)
        f1_list.append(f1)
        print("In this Run, Acc = ", acc, ", F1 = ", f1)
    # Print result
    print("===================== ACC =====================")
    for i in range(len(acc_list)):
        print(acc_list[i])
    print("=================== F1-score ==================")
    for i in range(len(f1_list)):
        print(f1_list[i])


