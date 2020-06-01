import torch
import torch.nn as nn
from torchvision import transforms
import sklearn
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import numpy as np
import scipy.io as sio
import os
import spectral
from torchsummary import summary
import copy
from modules import NT_Xent
import time
import csv
import argparse
from visdom import Visdom
apex = False
class ModifiedContrastive(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, c=16):
        super(ModifiedContrastive, self).__init__()
        self.classes = c

    def forward(self, x0, x1, y1, y2):
        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        dist_to_label=(dist-(y1-y2).abs_().float())/self.classes
        loss = (y1-y2==0).float() * dist_sq + (y1-y2!=0).float() * torch.pow(dist_to_label, 2)
        loss = torch.sum(loss)/ x0.size()[0]
        return loss
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))

        # euclidian distance
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
class ConvBNRelu3D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), padding=0,stride=1):
        super(ConvBNRelu3D,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.stride=stride
        self.conv=nn.Conv3d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm3d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x
class ConvBNRelu2D(nn.Module):
    def __init__(self,in_channels=1, out_channels=24, kernel_size=(51, 3, 3), stride=1,padding=0):
        super(ConvBNRelu2D,self).__init__()
        self.stride = stride
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.padding=padding
        self.conv=nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, stride=self.stride,padding=self.padding)
        self.bn=nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x= self.relu(x)
        return x

class HyperCLR(nn.Module):
    def __init__(self,channel,output_units,windowSize):
        # 调用Module的初始化
        super(HyperCLR, self).__init__()
        self.channel=channel
        self.output_units=output_units
        self.windowSize=windowSize
        self.conv1 = ConvBNRelu3D(in_channels=1,out_channels=8,kernel_size=(7,3,3),stride=1,padding=0)
        self.conv2 = ConvBNRelu3D(in_channels=8,out_channels=16,kernel_size=(5,3,3),stride=1,padding=0)
        self.conv3 = ConvBNRelu3D(in_channels=16,out_channels=32,kernel_size=(3,3,3),stride=1,padding=0)
        self.conv4 = ConvBNRelu2D(in_channels=32*(self.channel-12), out_channels=64, kernel_size=(3, 3), stride=1, padding=0)
        self.pool=nn.AdaptiveAvgPool2d((4, 4))
        self.projector = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,256),
        )
        self.fc=nn.Linear(1024,512)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(512, self.output_units)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape([x.shape[0],-1,x.shape[3],x.shape[4]])
        x = self.conv4(x)
        x = self.pool(x)
        x = x.reshape([x.shape[0], -1])
        h = self.projector(x)
        x=self.fc(x)
        x=self.relu1(x)
        z=self.fc2(x)
        return h, z
def train(encoder,Datapath1,Datapath2,PairLabelpath,Datapath,Labelpath,trans,epochs=50):
    contrast_data = PairDataset(Datapath1, Datapath2, PairLabelpath, trans)
    contrast_loader = torch.utils.data.DataLoader(dataset=contrast_data, batch_size=args.batch_size, shuffle=True,
                                                  drop_last=False)

    optim_contrast = torch.optim.Adam(encoder.parameters(), lr=5e-5, weight_decay=0.0005)
    contrast_criterion = ModifiedContrastive()

    classi_data = MYDataset(Datapath, Labelpath, trans)  # supervised part
    classi_loader = torch.utils.data.DataLoader(dataset=classi_data, batch_size=args.batch_size, shuffle=True)
    optim_classi = torch.optim.Adam(encoder.parameters(), lr=args.classi_lr, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = 10000
    best_model_wts = copy.deepcopy(encoder.state_dict())
    best_acc = 0
    for epoch in range(epochs):
        train_acc = 0
        epoch_contrastloss = 0
        epoch_classiloss=0
        print(" Epoch No  {} ".format(epoch))
        for step, (x_i, x_j, label_i, label_j) in enumerate(contrast_loader):

            x_i = x_i.cuda().float()
            x_j = x_j.cuda().float()
            label_i = label_i.cuda()
            label_j = label_j.cuda()
            encoder = encoder.cuda()
            encoder.train()
            # positive pair, with encoding
            h_i, z_i = encoder(x_i)
            h_j, z_j = encoder(x_j)

            contrast_loss = contrast_criterion(h_i, h_j, label_i, label_j)
            epoch_contrastloss += contrast_loss.item()
            optim_contrast.zero_grad()
            contrast_loss.backward(retain_graph=True)
            optim_contrast.step()
        for i, (data, label) in enumerate(classi_loader):  # supervised part
            data = data.cuda().float()
            label = label.cuda()
            h, z =encoder(data)
            classi_loss = criterion(z, label)
            epoch_classiloss += classi_loss.item()
            pred = torch.max(z, 1)[1]
            train_correct = (pred == label).sum()
            train_acc += train_correct.item()
            optim_classi.zero_grad()
            classi_loss.backward(retain_graph=True)
            optim_classi.step()
        print(
            'Train Loss: {:.6f}, Acc: {:.6f}, Contrast Loss: {:.6f}'.format(classi_loss / (len(classi_data)), train_acc / (len(classi_data)),contrast_loss / (len(contrast_data))))
        if (train_acc / (len(classi_data)) >= best_acc) and ((classi_loss / (len(classi_data))+contrast_loss / (len(contrast_data))) < best_loss):
            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc=train_acc / (len(classi_data))
            best_loss=(classi_loss / (len(classi_data))+contrast_loss / (len(contrast_data)))
    torch.save(best_model_wts, 'model.pth')
    return 0
def predict(model,Datapath,Labelpath):
    model.eval()
    model = model.cuda()
    test_data = MYDataset(Datapath,Labelpath,trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    prediction=[]
    for data, label in test_loader:
        data=data.cuda().float()
        h, out = model(data)
        for num in range(len(out)):
            prediction.append(np.array(out[num].cpu().detach().numpy()))
    return prediction
def SaveFeature(model,Datapath,Labelpath):
    model.eval()
    model = model.cuda()
    test_data = MYDataset(Datapath, Labelpath, trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    hFeature = []
    for data, label in test_loader:
        data = data.cuda().float()
        h, out = model(data)
        for num in range(len(h)):
            hFeature.append(np.array(h[num].cpu().detach().numpy()))
            np.save('hFeature.npy',hFeature)
    return 0
def evaluate(model,Datapath,Labelpath):
    model.eval()
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    test_data = MYDataset(Datapath,Labelpath,trans)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    score=np.zeros(2)
    train_loss=0.
    train_acc=0.
    index=0
    for data, label in test_loader:
        data=data.cuda().float()
        h,out = model(data)
        label = label.cuda()
        loss = criterion(out, label)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred ==label).sum()
        train_acc += train_correct.item()
    score[0]=train_loss/ (len(test_data))
    score[1]=train_acc/ (len(test_data))
    return score
def loadData(name):
    data_path = os.path.join(os.getcwd(), 'datasets')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    return data, labels
def splitTrainTestSet(X, y, testRatio, randomState=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    # patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]), dtype=np.float16)
    # patchesLabels = np.zeros((X.shape[0] * X.shape[1]), dtype=np.float16)
    patchesData=[]
    patchesLabels=[]
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):

            patch = zeroPaddedX[0:2*margin + 1, c - margin:c + margin + 1]
            patchesData.append(patch)
            patchesLabels.append(y[r - margin, c - margin])
        zeroPaddedX=np.delete(zeroPaddedX,0,axis=0)
    del zeroPaddedX
    if removeZeroLabels:
        indexList=[i for i in range(len(patchesLabels)) if patchesLabels[i] == 0]
        k=0
        for index in indexList:
            del patchesData[index-k]
            k+=1
        k=0
        for index in indexList:
            del patchesLabels[index-k]
            k+=1
        patchesLabels = [i - 1 for i in patchesLabels]
    return patchesData, patchesLabels
def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
def reports(model,Datapath, Labelpath, name):
    # start = time.time()
    y_pred = predict(model,Datapath,Labelpath)
    y_pred = np.argmax(np.array(y_pred), axis=1)
    # end = time.time()
    # print(end - start)
    Label=np.load(Labelpath).astype(int)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1', 'Brocoli_green_weeds_2', 'Fallow', 'Fallow_rough_plow',
                        'Fallow_smooth',
                        'Stubble', 'Celery', 'Grapes_untrained', 'Soil_vinyard_develop', 'Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk', 'Lettuce_romaine_5wk', 'Lettuce_romaine_6wk', 'Lettuce_romaine_7wk',
                        'Vinyard_untrained', 'Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

    classification = classification_report(Label, y_pred, target_names=target_names)
    oa = accuracy_score(Label, y_pred)
    confusion = confusion_matrix(Label, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(Label, y_pred)
    score = evaluate(model,Datapath,Labelpath)
    Test_Loss = score[0] * 100
    Test_accuracy = score[1] * 100
    return classification, confusion, Test_Loss, Test_accuracy, oa * 100, each_acc * 100, aa * 100, kappa
class PairDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath1,Datapath2,Labelpath,trans):
        # 1. Initialize file path or list of file names.
        self.data=np.load('Xtrain.npy')
        self.DataList1=np.load(Datapath1)
        self.DataList2 = np.load(Datapath2)
        self.LabelList=np.load(Labelpath)
        self.trans=trans
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data

        index=index
        num1=self.DataList1[index]

        Data=self.trans(self.data[num1].astype('float64'))
        Data=Data.view(-1,Data.shape[0],Data.shape[1],Data.shape[2])
        num2=self.DataList2[index]
        Data2=self.trans(self.data[num2].astype('float64'))
        Data2=Data2.view(-1,Data2.shape[0],Data2.shape[1],Data2.shape[2])
        Label=self.LabelList[index]
        return Data, Data2, Label[0], Label[1]
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.DataList1)
class MYDataset(torch.utils.data.Dataset):#需要继承data.Dataset
    def __init__(self,Datapath,Labelpath,transform):
        # 1. Initialize file path or list of file names.
        self.Datalist=np.load(Datapath)
        self.Labellist=(np.load(Labelpath)).astype(int)
        self.transform=transform
    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        #这里需要注意的是，第一步：read one data，是一个data

        index=index
        Data=self.transform(self.Datalist[index].astype('float64'))
        Data=Data.view(1,Data.shape[0],Data.shape[1],Data.shape[2])
        return Data ,self.Labellist[index]
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.Datalist)

dataset_names = ['IP', 'SA', 'PU']
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='IP', choices=dataset_names,
                    help="Dataset to use.")
parser.add_argument('--train',type=bool, default=1)
parser.add_argument('--perclass', type=float, default=300)
parser.add_argument('--device', type=str, default="cuda:0", choices=("cuda:0","cuda:1"))
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--classi_lr', type=float, default=3e-3)
args = parser.parse_args()
TRAIN =args.train
epoch=args.epoch
dataset = args.dataset
perclass=args.perclass
perclass=perclass/100

output_units = 9 if (dataset == 'PU' or dataset == 'PC') else 16
#IP 10249   PU 42776   SA  54129
if dataset=='IP':
    Total=10249
elif dataset=='PU':
    Total=42776
elif dataset=='SA':
    Total=54129
if perclass>1:
    perclass=int(perclass)
    test_ratio = Total-output_units*perclass
else:
    test_ratio =perclass
windowSize = 25
X, y = loadData(dataset)
K = X.shape[2]
K = 30 if dataset == 'IP' else 15
trans = transforms.Compose(transforms = [
    transforms.ToTensor(),
    transforms.Normalize(np.zeros(K),np.ones(K))
])
X0, pca = applyPCA(X, numComponents=K)
del X
X, y = createImageCubes(X0.astype('float16'), y, windowSize=windowSize)
def feature_normalize(data):
    mu = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data - mu)/std
X=feature_normalize(np.asarray(X).astype('float16'))
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y, test_ratio)
del X
np.save('Xtrain.npy',Xtrain)
np.save('ytrain.npy',ytrain)
np.save('Xtest.npy',Xtest)
np.save('ytest.npy',ytest)
del Xtrain, Xtest, ytest
datalist1=[]
datalist2=[]
labellist=[]

for order in range(len(ytrain)):
    for k in range(len(ytrain)):
        if not order==k:
            y=[ytrain[order],ytrain[k]]
            datalist1.append(order)
            datalist2.append(k)
            labellist.append(y)
np.save('TrainData1.npy',datalist1)
np.save('TrainData2.npy',datalist2)
np.save('TrainLabel.npy',labellist)
del datalist1, datalist2, labellist
model=HyperCLR(channel=K,output_units=output_units,windowSize=windowSize)
model=model.cuda()
summary(model,(1,K,windowSize,windowSize))
train_start_time=time.time()
if TRAIN:
    Datapath1 = 'TrainData1.npy'
    Datapath2 = 'TrainData2.npy'
    PairLabelpath = 'TrainLabel.npy'
    Datapath='Xtrain.npy'
    Labelpath='ytrain.npy'
    train(model,Datapath1,Datapath2,PairLabelpath,Datapath,Labelpath,trans,epochs=epoch)
train_end_time=time.time()
#Testing
model.load_state_dict(torch.load('model.pth'))
Datapath='Xtest.npy'
Labelpath='ytest.npy'
test_start_time=time.time()
Y_pred = predict(model, Datapath, Labelpath)
Y_pred = np.argmax(np.array(Y_pred), axis=1)
classification = classification_report(np.load(Labelpath).astype(int), Y_pred)
test_end_time=time.time()
print(classification)
print()
print('training time:',train_end_time-train_start_time,'s')
print()
print('testing time:',test_end_time-test_start_time,'s')

classification, confusion, Test_loss, Test_accuracy, oa, each_acc, aa, kappa = reports(model,Datapath, Labelpath, dataset)
add_info=[dataset,perclass,oa,kappa,aa,train_end_time-train_start_time,test_end_time-test_start_time]
csvFile = open("siamse_experiments.csv", "a")
writer = csv.writer(csvFile)
writer.writerow(add_info)
csvFile.close()
SaveFeature(model,Datapath,Labelpath)