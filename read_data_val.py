
from __future__ import print_function

import torch
import torch.utils.data as data
import pickle




path = 'D:\\DeskTop\\MVRL_codes_swt\\code\\AVCMRL\\multi-view_data\\'


# RBP 3 views
class RBPDATASET(data.Dataset):
    def __init__(self, set_name='train'):
        self.processed_folder = path
        self.set_name = set_name
        self.train_file = 'RBP3viewtrain.pkl'
        self.validation_file = 'RBP3valcheck.pkl'
        self.test_file = 'RBP3viewtest.pkl'

        if self.set_name == 'train':
            with open(self.processed_folder + self.train_file, 'rb') as fp:
                self.view1_data, self.view2_data, self.view3_data, self.labels = pickle.load(fp)

        elif self.set_name == 'validation':
            with open(self.processed_folder + self.validation_file, 'rb') as fp:
                self.view1_data, self.view2_data, self.view3_data, self.labels = pickle.load(fp)

        elif self.set_name == 'test':
            with open(self.processed_folder + self.test_file, 'rb') as fp:
                self.view1_data, self.view2_data, self.view3_data, self.labels = pickle.load(fp)


        if len(self.labels.shape) > 1:
                self.labels = self.labels.squeeze()


        assert len(self.view1_data) == len(self.labels), "No Match！"

    def __getitem__(self, index):
        view1 = self.view1_data[index]
        view2 = self.view2_data[index]
        view3 = self.view3_data[index]
        target = self.labels[index]
        return view1, view2, view3, target

    def __len__(self):
        return len(self.view1_data)


# RBP 3 views
train_data = RBPDATASET(set_name='train')
# #val_data = RBPDATASET(set_name='validation')
test_data = RBPDATASET(set_name='test')




print("================train_data======================")
print("view1:", len(train_data.view1_data),train_data.view1_data.shape)
print("view2:", len(train_data.view2_data),train_data.view2_data.shape)
print("view3:", len(train_data.view3_data),train_data.view3_data.shape)
print("label:", len(train_data.labels),train_data.labels.shape)

# # print("================validation_data=================")
# # print("view1:", len(val_data.view1_data),val_data.view1_data.shape)
# # print("view2:", len(val_data.view2_data),val_data.view2_data.shape)
# # print("view3:", len(val_data.view3_data),val_data.view3_data.shape)
# # print("label:", len(val_data.labels),val_data.labels.shape)

print("================test_data=================")
print("view1:", len(test_data.view1_data),test_data.view1_data.shape)
print("view2:", len(test_data.view2_data),test_data.view2_data.shape)
print("view3:", len(test_data.view3_data),test_data.view3_data.shape)

print("label:", len(test_data.labels),test_data.labels.shape)



# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=256, shuffle=True)
#
# for train_iteration_index, train_data in enumerate(train_loader):
#     page_input, link_input, view3_input, train_labels = train_data
#
#     #train_labels = torch.squeeze(train_labels)
#     print(page_input.shape, link_input.shape, view3_input.shape, train_labels.shape)












