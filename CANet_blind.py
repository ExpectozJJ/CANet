import argparse, sys, time, random, torch, re, joblib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def dataset_list(filename):
    dataset = []
    fp = open(filename)
    for line in fp:
        line_split = re.split(',|\n', line)
        dataset.append(line_split[:-1])
    fp.close()
    return dataset

def normalize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X)

def RMSE(ypred, yexact):
    return torch.sqrt(torch.sum((ypred-yexact)**2)/ypred.shape[0])

def PCC(ypred, yexact):
    a = ypred.cpu().numpy().ravel()
    b = yexact.cpu().numpy().ravel()
    pcc = stats.pearsonr(a, b)
    return pcc

class TopLapNet(Dataset):
    def __init__(self, X, y, transforms=transforms.Compose([])):
        self.X = X
        self.labels = y
        self.transforms = transforms

    def __getitem__(self, index):
        X_array2tensor = torch.from_numpy(self.X[index]).float()
        if self.transforms is not None:
            X_array2tensor = self.transforms(X_array2tensor)
        return (X_array2tensor, self.labels[index])

    def __len__(self):
        return self.X.shape[0]

class MultitaskModule(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MultitaskModule, self).__init__()

        # input layer and initialize weights
        self.input_layer = nn.Linear(D_in, H[0], bias=True)
        nn.init.xavier_uniform_(self.input_layer.weight)

        # hiden layer and initialize weights
        self.hiden_layers = nn.ModuleList([nn.Linear(H[i], H[i+1], bias=True) 
                                           for i in range(len(H)-1)])
        for hiden_layer in self.hiden_layers:
            nn.init.xavier_uniform_(hiden_layer.weight)

        # output layer and initialize weights
        # self.output_layer = nn.Linear(H[-1], D_out, bias=True)
        self.output_layer = nn.Linear(H[-1], D_out, bias=True)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, X):
        X = F.relu(self.input_layer(X))
        for hiden_layer in self.hiden_layers:
            X = F.relu(hiden_layer(X))
        y = self.output_layer(X)
        #y = -12 + 12*F.tanh(self.output_layer(X))
        return y

def train(model, device, train_loader, criterion, optimizer):
    model.train() # tells your model that you are training the model
    for (data, target) in train_loader:
        # move tensor to computing device ('gpu' or 'cpu')
        data, target = data.to(device), target.to(device).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(data).view(-1, 1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader, epoch):
    model.eval() # tell that you are testing, == model.train(model=False)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data).view(-1, 1)
            test_loss = F.mse_loss(output, target, reduction='sum').item()
            test_loss /= len(test_loader.dataset)
            pcc = PCC(output, target)[0]
            rmse = RMSE(output, target)
            print('Epoch: %d, test_loss: %.4f, RMSE: %.4f, PCC: %.4f'%(epoch, test_loss, rmse, pcc))
            return output.cpu().numpy(), target.cpu().numpy()

tic = time.perf_counter()

parser = argparse.ArgumentParser(description='CANet')
parser.add_argument('--dataset', type=str, default='S2648',
                    help='input batch size for training (default: 50)')
parser.add_argument('--datatype', type=str, default='all',
                    help='input batch size for training (default: 50)')
parser.add_argument('--batch_size', type=int, default=50,
                    help='input batch size for training (default: 50)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='SGD weight decay (default: 0)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, 
                    help='how many batches to wait before logging training status')
parser.add_argument('--layers', type=str, default='15000,15000,15000,15000,15000,15000',
                    help='neural network layers and neural numbers')
parser.add_argument('--nlayer', type=int, default=6,
                    help='number of neural network layers')
args = parser.parse_args()
print(args)
torch.manual_seed(args.seed)

# setup device cuda or cpu
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#device = torch.device("cpu")

# protein stability change upon mutation features and labels
if args.datatype == 'aux':
    X_val = np.load('./S2648/X_'+args.dataset+'_aux.npy')
elif args.datatype == 'FRI':
    X_val = np.load('./S2648/X_'+args.dataset+'_FRI.npy')
elif args.datatype == 'PH0':
    X_val = np.load('./S2648/X_'+args.dataset+'_PH0.npy')
elif args.datatype == 'PH12':
    X_val = np.load('./S2648/X_'+args.dataset+'_PH12.npy')
elif args.datatype == 'ESM':
    X_val = np.load('./S2648/X_'+args.dataset+'_ESM.npy')
elif args.datatype == 'Lap':
    X_val = np.load('./S2648/X_'+args.dataset+'_Lap_b.npy')
elif args.datatype == 'all':
    X_val1 = np.load('./S2648/X_'+args.dataset+'_aux.npy')
    X_val2 = np.load('./S2648/X_'+args.dataset+'_FRI.npy')
    #X_val3 = np.load('./S2648/X_'+args.dataset+'_PH0.npy')
    #X_val4 = np.load('./S2648/X_'+args.dataset+'_PH12.npy')
    X_val3 = np.load('./S2648/X_'+args.dataset+'_SRcurves.npy')
    #X_val3 = np.load('./S2648/X_S2648_SRrates.npy')
    X_val4 = np.load('./S2648/X_'+args.dataset+'_SRfcurves.npy')

    X_val5 = np.load('./S2648/X_'+args.dataset+'_ESM.npy')
    
    #X_val6 = np.load('./S2648/X_'+args.dataset+'_Lap_b.npy')
    X_val = np.concatenate((X_val1, X_val2), axis=1)
    X_val = np.concatenate((X_val,  X_val3), axis=1)
    X_val = np.concatenate((X_val,  X_val4), axis=1)
    X_val = np.concatenate((X_val,  X_val5), axis=1)
    #X_val = np.concatenate((X_val,  X_val6), axis=1)

X_val = normalize(X_val)[::2]
#normalizer1 = joblib.load('model/normalizer_mini_alphafold.pkl')
#X_val = normalizer1.transform(X_val_skempi2)
#normalizer2 = joblib.load('model/normalizer_Lap_ESM_mini_alphafold.pkl')
#X_val_Lap_ESM = normalizer2.transform(X_val_skempi2_Lap_ESM)

#X_val1 = np.concatenate((X_val[:, :759], X_val[:, 759+648:]), axis=1)
#X_val = np.concatenate((X_val1, X_val_Lap_ESM), axis=1)
y_val = np.load(f'./S2648/Y_{args.dataset}.npy').reshape((-1, 1))[::2]
print('The data shape', X_val.shape, ', label size', y_val.shape)

data = dataset_list(f'./S2648/S350.txt')
all_data = dataset_list(f'./S2648/S2648.txt')
train_idx = list(range(len(all_data)))
test_idx = []
for i in range(len(data)):
    ilist = data[i]
    PDBid, Antibody, Chain, resWT, resID, resMT, pH, ddG = ilist[0], ilist[1], ilist[2], ilist[3], ilist[4], ilist[5], ilist[6], float(ilist[7])
    flag = False
    for j in range(len(all_data)):
        ilist2 = all_data[j]
    
        PDBid2, Antibody2, Chain2, resWT2, resID2, resMT2, pH2, ddG2 = ilist2[0], ilist2[1], ilist2[2], ilist2[3], ilist2[4], ilist2[5], ilist2[6], float(ilist2[7])
        #print(ilist2)
        if PDBid2 == PDBid and Antibody == Antibody2 and Chain2 == Chain and resWT == resWT2 and resID == resID2 and resMT == resMT2 and pH == pH2:
            test_idx.append(j)
            flag = True 
            break 
    
    if flag == False:
        print(ilist)

train_idx = list(set(train_idx)-set(test_idx))
print(len(test_idx), len(train_idx))
X_train, y_train = X_val[train_idx], y_val[train_idx]
X_test, y_test = X_val[test_idx], y_val[test_idx]

hiden_layer = [int(i) for i in args.layers.split(',')]

#y_pred = np.zeros(len(y_test))
#y_real = np.zeros(len(y_test))
kwargs = {'shuffle': True, 'num_workers': 1, 'pin_memory': True} if use_cuda else {'shuffle': True}
#kwargs = {'shuffle': True}
#kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)

for ii in range(10):
    #for idx, (train_idx, test_idx) in enumerate(kf.split(X_val)):
    # setup dataloader
    #X_train, X_test = X_val[train_idx], X_val[test_idx]
    #y_train, y_test = y_val[train_idx], y_val[test_idx]
    train_dataset = TopLapNet(X_train, y_train)
    test_dataset  = TopLapNet(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=len(test_idx), **kwargs)

    model = MultitaskModule(X_val.shape[1], hiden_layer, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum, 
                            weight_decay=args.weight_decay)
    lr_adjust = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1, last_epoch=-1)
    for epoch in range(args.epochs):
        train(model, device, train_loader, criterion, optimizer)
        if epoch%args.log_interval == 0:
            print('epoch %d >>>>>>>>>>>>>>>>>>>>>>>>'%epoch)
            test(model, device, test_loader, epoch)
            print(f'train data shape {X_train.shape}')
            print(f'test data shape {X_test.shape}')
        lr_adjust.step()
    
    print('epoch %d >>>>>>>>>>>>>>>>>>>>>>>>'%epoch)
    test(model, device, test_loader, epoch)

    model.to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    model.eval()
    with torch.no_grad():
        ypred = model(X_test)[:, 0].view(-1, 1).cpu().numpy().ravel()

    y_pred = np.reshape(ypred, len(ypred))
    y_real = np.reshape(y_test, len(y_test))

    #ypred = ypred[::2]
    fp = open(f'./S2648/{args.dataset}_blind_{ii}_CAnet.txt', 'w+')
    for i in range(len(y_real)):
        fp.write(f'{y_pred[i]} {y_real[i]}\n')
    fp.close()
    pcc = stats.pearsonr(y_pred, y_real)[0]
    rmse = np.sqrt(mean_squared_error(y_pred, y_real))
    toc = time.perf_counter()
    print('RMSE: %.3f, Rp: %.4f\nElapsed time: %.1f [min]'%(rmse, pcc, (toc-tic)/60))

