import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from Model import ActionPredictor
from Dataset import PredictorData

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHES = 20
SEEN_DATA_PATH = "./seen_data/"
UNSEEN_DATA_PATH = "./unseen_data/"
JSON_NAME = "success"
PRINT_ITER = 100
TRAIN_RATIO = 0.95
THRESHOLD = 0.7
WEIGHT_DECAY=0.001

net = ActionPredictor(2).cuda()
optimizer = torch.optim.Adam(net.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)
full_dataset = PredictorData(SEEN_DATA_PATH, JSON_NAME)
unseen_dataset = PredictorData(UNSEEN_DATA_PATH, JSON_NAME)
num_of_sample = len(full_dataset)
print("TOTAL SAMPLE NUMBER:{:d} \n TRAIN SAMPLE NUM:{:d} \n VAL SAMPLE NUM:{:d}".format(num_of_sample, int(num_of_sample*TRAIN_RATIO), int(num_of_sample*(1-TRAIN_RATIO))))
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [int(num_of_sample*TRAIN_RATIO), num_of_sample - int(num_of_sample*TRAIN_RATIO)])
train_data = DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle = True, drop_last=True)
val_data = DataLoader(val_dataset, batch_size= BATCH_SIZE, shuffle = False, drop_last=True)
unseen_val_data = DataLoader(unseen_dataset, batch_size= BATCH_SIZE, shuffle = False, drop_last=True)
loss_function = nn.BCEWithLogitsLoss()#FocalLoss()
for i in range(EPOCHES):
    for j, data in enumerate(train_data):
        feature = data[0].cuda()
        label = data[1].cuda()
        prediction = net(feature)
        optimizer.zero_grad()
        loss = loss_function(prediction, label)
        loss.backward()
        optimizer.step()
        if j % PRINT_ITER == 0:
            print(loss.detach().cpu())
    total_num = 0
    correct_num = 0
    positive_num = 0
    negative_num = 0
    positive_correct = 0
    negative_correct = 0
    net.eval()
    for j, data in enumerate(val_data):
        feature = data[0].cuda()
        label = data[1].cuda()
        positive_num += (label==1).int().sum()
        negative_num += (label==0).int().sum()
        prediction = net(feature)
        prediction = prediction.detach()
        total_num += prediction.size(0)
        correct = ((prediction > THRESHOLD).int() == label).int()
        positive_correct += (correct * (label==1).int()).sum()
        negative_correct += (correct * (label==0).int()).sum()
        correct_num += correct.sum()
    print("-----------------SEEN DATA EVALUATION-------------------")
    print("Evaluation Acc{:.2f}".format(correct_num/total_num))
    print("Positive AP{:.2f}".format(positive_correct/positive_num))
    print("Negative AP{:.2f}".format(negative_correct/negative_num))
    total_num = 0
    correct_num = 0
    positive_num = 0
    negative_num = 0
    positive_correct = 0
    negative_correct = 0
    for j, data in enumerate(unseen_val_data):
        feature = data[0].cuda()
        label = data[1].cuda()
        positive_num += (label==1).int().sum()
        negative_num += (label==0).int().sum()
        prediction = net(feature)
        prediction = prediction.detach()
        total_num += prediction.size(0)
        correct = ((prediction > THRESHOLD).int() == label).int()
        positive_correct += (correct * (label==1).int()).sum()
        negative_correct += (correct * (label==0).int()).sum()
        correct_num += correct.sum()
    print("-----------------UNSEEN DATA EVALUATION-------------------")
    print("Evaluation Acc{:.2f}".format(correct_num/total_num))
    print("Positive AP{:.2f}".format(positive_correct/positive_num))
    print("Negative AP{:.2f}".format(negative_correct/negative_num))

    net.train()
torch.save(net, "checkpoint_end_{:d}".format(i + 1))
