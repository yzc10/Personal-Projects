"""
Guideline of your submission of HW3.
If you have any questions in regard to submission,
please contact TA: Ma Zhiyuan <e0983565@u.nus.edu>
"""

from os import chdir,walk,listdir,makedirs,path,remove
import numpy as np
import math
import shutil
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support,balanced_accuracy_score
import torch
from torch.utils.data import Dataset,TensorDataset, DataLoader,RandomSampler
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, resnet34, resnet50, vgg16, alexnet, resnext50_32x4d, mnasnet1_0
from torchvision import transforms
import cv2
import warnings
import copy
import gc

###################################### Key objects #####################################################################
if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'
labels = {'bedroom':1, 'Coast':2, 'Forest':3, 'Highway':4, 'industrial':5, 'Insidecity':6, 'kitchen':7, 'livingroom':8, 'Mountain':9, 'Office':10, 'OpenCountry':11, 'store':12, 'Street':13, 'Suburb':14, 'TallBuilding':15}
model_config = {
    'lr': 4e-4,
    'batch_size':32,
    'n_epochs':200,
    'patience':20,
    'step_size':10, #scheduler
    'gamma':0.7 #scheduler
}

# Custom ResNet34
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    # Original: 64 64 128 256 512
    def __init__(self, block, layers, num_classes = 15):
        super(ResNet, self).__init__()
        self.inplanes = 10
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 10, kernel_size = 5, stride = 2, padding = 3),
                        nn.BatchNorm2d(10),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 20, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 30, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 40, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 40, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(5, stride=1)
        self.fc = nn.Linear(360, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Custom CNN
class CNN(nn.Module):
    def __init__(self,attention=False):
        super(CNN,self).__init__()
        
        self.cnn = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(1,10,kernel_size=7,stride=1,padding=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            # Conv layer 2
            nn.Conv2d(10,20,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            # Conv layer 3
            nn.Conv2d(20,25,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            # # Conv layer 4
            nn.Conv2d(25,25,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            # # Conv layer 5
            nn.Conv2d(25,25,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.ff = nn.Sequential(
            nn.Linear(25*6*6,15),
        )
        self.attention=attention
        if self.attention:
            self.attn = nn.MultiheadAttention(6*6,1,batch_first=True) #hidden_size,num_heads. Change to 12*12 for 4 conv layers

    def forward(self,x):
        x = self.cnn(x)
        x = x.view(x.size(0),x.size(1),-1) # B * F * (w*h)
        if self.attention:
            context,_ = self.attn(x,x,x) # context: B * F * (w*h)
            x = torch.reshape(context,(context.size(0),-1))
        else:
            x = torch.reshape(x,(x.size(0),-1))
        x = self.ff(x)
        return x
    
class MyDataset(Dataset):
    def __init__(self, data, target, transform=True):
        self.data = data
        self.target = target
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)      
        y = self.target[index]            
        return x, y

###################################### Subroutines #####################################################################
"""
Example of subroutines you might need. 
You could add/modify your subroutines in this section. You can also delete the unnecessary functions.
It is encouraged but not necessary to name your subroutines as these examples. 
"""

def generate_train_test_sets(main_path,test_pct=0.2):
    cat_folders = []
    for (_,dirname,_) in walk(main_path):
        cat_folders.extend(dirname)
        break
    for cat_name in tqdm(cat_folders):
        img_list = listdir(path.join(main_path,cat_name))
        rng = np.random.default_rng(seed=42)
        train_num = int(len(img_list) * (1-test_pct))
        train_img_list = rng.choice(img_list,train_num,replace=False).tolist()
        test_img_list = list(set(img_list).difference(set(train_img_list)))
        type_to_list = {'train':train_img_list,'test':test_img_list}
        for type in type_to_list:
            target_path = path.join(main_path,path.join(type,cat_name))
            if not path.exists(target_path):
                makedirs(target_path)
            for img_file in type_to_list[type]:
                shutil.copyfile(path.join(main_path,path.join(cat_name,img_file)),
                                path.join(target_path,img_file))

def resize_img(img,tgt_x=200,tgt_y=200):
    import math
    x,y = img.shape
    scale = max(tgt_x/x,tgt_y/y)
    img1 = cv2.resize(img,(math.ceil(scale*y),math.ceil(scale*x)),interpolation=cv2.INTER_LINEAR)
    x1,y1 = img1.shape
    dim_map = {0:[x1,tgt_x],1:[y1,tgt_y]}
    for dim in dim_map:
        dif = dim_map[dim][0] - dim_map[dim][1]
        if dif >= 2: 
            if dim == 0:
                img1 = img1[dif//2:x1-(dif-(dif//2)),:]
            elif dim == 1:
                img1 = img1[:,dif//2:y1-(dif-(dif//2))]
        elif dif == 1:
            if dim == 0:
                img1 = img1[0:x1-1,:]
            elif dim == 1:
                img1 = img1[:,0:y1-1]
    return img1

def load_images(train_data_dir):
    if 'train' not in train_data_dir:
        raw = True
    else:
        raw = False
    cat_folders = []
    for (_,dirnames,_) in walk(train_data_dir):
            cat_folders.extend(dirnames)
            break
    if raw==True:
        exclude_folders = ['test','train','train_aug','train_aug2']
        cat_folders = [f for f in cat_folders if f not in exclude_folders]
    X_arr0 = []
    y_arr = []
    max_x = -np.inf # To gauge resizing dimensions
    min_x = np.inf
    max_y = -np.inf
    min_y = np.inf
    for cat in tqdm(cat_folders):
        cat_path = path.join(train_data_dir,cat)
        img_list = listdir(cat_path)
        for img in img_list:
            img_mat = np.array(cv2.imread(path.join(cat_path,img),-1), dtype=np.float32)
            X_arr0.append(img_mat)
            y_arr.append(labels[cat])
            x,y = img_mat.shape
            if x > max_x: max_x = x
            elif x < min_x: min_x = x
            if y > max_y: max_y = y
            elif y < min_y: min_y = y
    return X_arr0, y_arr

# Custom transformations
def augment_dataset(source_data_dir,aug_data_dir):
    if path.exists(aug_data_dir):
        shutil.rmtree(aug_data_dir)
    makedirs(aug_data_dir)
    X_aug, y_aug = [], []
    X_arr0, y_arr = load_images(source_data_dir)
    for i in tqdm(range(len(X_arr0))):
        im0 = X_arr0[i]
        X_aug.append(im0)
        # # Zoom and Crop:
        # im_ups = np.zeros(np.array(np.shape(im0))*2)
        # for k in range(2):
        #     for n in range(2):
        #         im_ups[k::2,n::2] = im0
        # im_ups1 = im_ups[100:300,100:300]
        # X_aug.append(im_ups1)
        
        # # Flip (Vertical/Horizontal):
        # im_flip1 = cv2.flip(im0,0)
        # X_aug.append(im_flip1)
        im_flip2 = cv2.flip(im0,1)
        X_aug.append(im_flip2)
        
        # # Rotate x 1:
        # mat = cv2.getRotationMatrix2D((im0.shape[1]//2,im0.shape[0]//2),10,1) # centre, degree, scale
        # im_rotate1 = cv2.warpAffine(im0,mat,(200,200)) # Hard-coded img size
        # X_aug.append(im_rotate1)
        # im_rotate2 = cv2.rotate(im0,cv2.ROTATE_90_COUNTERCLOCKWISE)
        # X_aug.append(im_rotate2)
        
        # Augment y:
        y_aug.extend([y_arr[i]]*2) # Hard-coded aug count
    y_aug = np.array(y_aug) # Need this for np.argwhere
    for cat in tqdm(labels):
        cat_path = path.join(aug_data_dir,cat)
        makedirs(cat_path)
        idx = labels[cat]
        img_idx = np.argwhere(y_aug==idx)
        for i in range(len(img_idx)):
            cv2.imwrite(path.join(cat_path,f'Result_{i}.jpg'),np.uint8(np.clip(np.around(X_aug[img_idx[i][0]],decimals=0),0,255)))    

# # Reference for upsampling with bicubic interpolation: https://www.geeksforgeeks.org/python-opencv-bicubic-interpolation-for-resizing-image/
# Interpolation kernel 
def u(s, a): 
    if (abs(s) >= 0) & (abs(s) <= 1): 
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
        
    elif (abs(s) > 1) & (abs(s) <= 2): 
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a 
    return 0

# Padding 
def padding(img, H, W): 
    zimg = np.zeros((H+4, W+4)) 
    zimg[2:H+2, 2:W+2] = img 
      
    # Pad the first/last two col and row 
    zimg[2:H+2, 0:2] = img[:, 0:1] 
    zimg[H+2:H+4, 2:W+2] = img[H-1:H, :] 
    zimg[2:H+2, W+2:W+4] = img[:, W-1:W] 
    zimg[0:2, 2:W+2] = img[0:1, :] 
      
    # Pad the missing eight points 
    zimg[0:2, 0:2] = img[0, 0]
    zimg[H+2:H+4, 0:2] = img[H-1, 0] 
    zimg[H+2:H+4, W+2:W+4] = img[H-1, W-1] 
    zimg[0:2, W+2:W+4] = img[0, W-1] 
      
    return zimg 

# Bicubic operation 
def bicubic(img, ratio, a): 
  
    # H = Height, W = weight
    H, W = img.shape 
    img = padding(img, H, W) 
  
    # Create new image 
    dH = math.floor(H*ratio) 
    dW = math.floor(W*ratio) 
    dst = np.zeros((dH, dW)) 
    h = 1/ratio 
    inc = 0
  
    for j in range(dH): 
        for i in range(dW): 

            # Getting the coordinates of nearby values 
            x, y = i * h + 2, j * h + 2

            x1 = 1 + x - math.floor(x) 
            x2 = x - math.floor(x) 
            x3 = math.floor(x) + 1 - x 
            x4 = math.floor(x) + 2 - x 

            y1 = 1 + y - math.floor(y) 
            y2 = y - math.floor(y) 
            y3 = math.floor(y) + 1 - y 
            y4 = math.floor(y) + 2 - y 

            # Considering all nearby 16 values 
            mat_l = np.matrix([[u(x1, a), u(x2, a), u(x3, a), u(x4, a)]]) 
            mat_m = np.matrix([[img[int(y-y1), int(x-x1)], 
                                img[int(y-y2), int(x-x1)], 
                                img[int(y+y3), int(x-x1)], 
                                img[int(y+y4), int(x-x1)]], 
                                [img[int(y-y1), int(x-x2)], 
                                img[int(y-y2), int(x-x2)], 
                                img[int(y+y3), int(x-x2)], 
                                img[int(y+y4), int(x-x2)]], 
                                [img[int(y-y1), int(x+x3)], 
                                img[int(y-y2), int(x+x3)], 
                                img[int(y+y3), int(x+x3)], 
                                img[int(y+y4), int(x+x3)]], 
                                [img[int(y-y1), int(x+x4)], 
                                img[int(y-y2), int(x+x4)], 
                                img[int(y+y3), int(x+x4)], 
                                img[int(y+y4), int(x+x4)]]]) 
            mat_r = np.matrix( 
                [[u(y1, a)], [u(y2, a)], [u(y3, a)], [u(y4, a)]]) 
            dst[j, i] = np.dot(np.dot(mat_l, mat_m), mat_r) 
    return dst 

# Generate upsampled images
def generate_resized_dataset(source_data_dir,target_data_dir):
    if path.exists(target_data_dir):
        shutil.rmtree(target_data_dir)
    makedirs(target_data_dir)
    ratio = 2
    a = -0.5
    X_arr0, y_arr = load_images(source_data_dir)
    X_arr0 = np.array(X_arr0)
    y_arr = np.array(y_arr) # Crucial

    for cat in tqdm(labels):
        cat_path = path.join(target_data_dir,cat)
        makedirs(cat_path)
        idx = labels[cat]
        img_idx = np.argwhere(y_arr==idx)
        for i in range(len(img_idx)):
            resized_img = bicubic(X_arr0[img_idx[i][0]],ratio,a)
            img_idx = np.argwhere(y_arr==idx)
            cv2.imwrite(path.join(cat_path,f'Result_{i}.jpg'),np.uint8(np.clip(np.around(resized_img,decimals=0),0,255)))

# Helper functions for model training
def convert_y(y):
    y_nn = np.zeros([y.shape[0],15])
    for i in range(len(y)):
        y_nn[i,y[i]-1] = 1
    return y_nn

def to_tensor(arr):
    arr = torch.from_numpy(arr).to(device).type(torch.FloatTensor)
    return arr

def save_checkpoint(state,filename='./checkpoint.tar'):
    if path.isfile(filename):
        remove(filename)
    torch.save(state,filename)

def build_model(model_type):
    if model_type == 'cnn':
        model = CNN().to(device)
    elif model_type == 'custom_res':
        model = ResNet(ResidualBlock,[3,4,6,3]).to(device)
    elif model_type == 'res18':
        model = resnet18(
            num_classes = 15
        ).to(device)
        model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False).to(device)
    elif model_type == 'res34':
        model = resnet34(
            num_classes = 15
        ).to(device)
        model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False).to(device)
    elif model_type == 'res50':
        model = resnet50(
            num_classes = 15
        ).to(device)
        model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False).to(device)    
    elif model_type == 'vgg16':
        model = vgg16(
            num_classes = 15
        ).to(device)
        model.features[0] = nn.Conv2d(1,64,kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False).to(device)
    elif model_type == 'alex':
        model = alexnet(
            num_classes = 15
        ).to(device)
        model.features[0] = nn.Conv2d(1,64,kernel_size=(11,11),stride=(4,4),padding=(2,2)).to(device)
    elif model_type == 'resnext':
        model = resnext50_32x4d(
            num_classes = 15
        ).to(device)
        model.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False).to(device)
    elif model_type == 'mnas':
        model = mnasnet1_0(
            num_classes=15
        ).to(device)
        model.layers[0] = nn.Conv2d(1,32,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False).to(device)
    return model

###################################### Main train and test Function ####################################################
"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the 
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training 
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test, 
computing accuracy, etc.
"""

def train(train_data_dir,config,model_type,model_dir=None,transform=True,**kwargs):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
    # # Load images
    X_arr0, y_arr0 = load_images(train_data_dir)
    
    # # Prepare data arrays
    X_arr = []
    for i in tqdm(range(len(X_arr0))):
        img_resized = resize_img(X_arr0[i])
        X_arr.append(img_resized)
    X_arr = np.array(X_arr)/255.0
    y_arr = np.array(y_arr0)
    X_arr = np.expand_dims(X_arr,axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X_arr,y_arr,test_size=0.2,random_state=42)
    # X_train = np.concatenate((X_train,X_train)) # To expand dataset
    # y_train = np.concatenate((y_train,y_train)) # To expand dataset
    X_train, X_val = to_tensor(X_train),to_tensor(X_val)
    y_train, y_val_nn = to_tensor(convert_y(y_train)),to_tensor(convert_y(y_val))
    
    # Configure model
    lr = config['lr']
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    loss_fn = nn.CrossEntropyLoss()
    if transform == False:
        dataset = TensorDataset(X_train.to(device),y_train.to(device))
    else:
        transforms_set = transforms.Compose([
            # transforms.RandomResizedCrop((200,200),scale=(0.5,1.0)),
            transforms.Resize(200),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(0,30))
        ])
        dataset = MyDataset(X_train.to(device),y_train.to(device),transform=transforms_set)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset,batch_size=batch_size,sampler=sampler)
    model = build_model(model_type)    
    optimizer = optim.Adam(model.parameters(),lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=config['step_size'],gamma=config['gamma'])
    
    # Train & evaluate
    best_loss = float('inf')
    best_epoch = 0
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for X_batch,y_batch in loader:
            y_pred_batch = model(X_batch)
            loss = loss_fn(y_pred_batch,y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        save_checkpoint({
            'epoch':(epoch+1),
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict(),
        })
        model.eval()
        with torch.no_grad():
            y_pred_nn = model(X_val.to(device))
            y_pred = torch.argmax(y_pred_nn,1).detach().cpu()+1
            acc = accuracy_score(y_val,y_pred)
            bal_acc = balanced_accuracy_score(y_val,y_pred)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                prec,recall,f1,_ = precision_recall_fscore_support(y_val,y_pred,average='weighted')
            val_loss = loss_fn(y_pred_nn,y_val_nn.to(device))
            tqdm.write(f'Epoch {epoch+1}: val loss={val_loss}, accuracy={acc}, balanced accuracy={bal_acc}, precision={prec}, recall={recall}, fscore={f1}')
            if val_loss < best_loss:
                best_loss = val_loss
                best_acc = acc
                best_recall = recall
                best_f1 = f1
                best_epoch = epoch + 1
                best_model_weights = copy.deepcopy(model.state_dict())
                patience = config['patience']
            else:
                patience -= 1
                if patience == 0: 
                    print(f'Early stopping triggered at epoch {epoch+1}. Best model is from epoch {best_epoch}.')
                    print(f'Key scores from the best epoch are: accuracy={best_acc}, recall={best_recall}, f1={best_f1}.')
                    break
        scheduler.step()
    torch.save(best_model_weights,model_dir)
    return best_acc

def test(test_data_dir, model_dir, model_type,**kwargs):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """
    model = build_model(model_type)
    model.load_state_dict(torch.load(model_dir))
    model = model.to(device)
    X_arr0, y_arr0 = load_images(test_data_dir)
    X_arr = []
    for i in tqdm(range(len(X_arr0))):
        img_resized = resize_img(X_arr0[i])
        X_arr.append(img_resized)
    X_arr = np.array(X_arr)/255.0
    y_arr = to_tensor(np.array(y_arr0))
    X_arr = to_tensor(np.expand_dims(X_arr,axis=1))
    y_pred_nn = model(X_arr.to(device))
    y_pred = torch.argmax(y_pred_nn,1).detach().cpu()+1
    acc = accuracy_score(y_arr,y_pred)
    return acc


if __name__ == '__main__':
    import argparse
    
    # # [Uncomment if required] Set path variable:
    # p = r'C:\YZC\NUS\Semester 2\DSA5203 Visual Data Processing and Interpretation\Assignments\assignment_3'
    # chdir(p)

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='model.pkl', help='the pre-trained model')
    opt = parser.parse_args()
    model_type = 'res18' # Final model selection
    gc.collect()
    if opt.phase == 'train':
        training_accuracy = train(train_data_dir=opt.train_data_dir,model_type=model_type,config=model_config,model_dir=opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir,model_type=model_type)
        print(testing_accuracy)
       
    # # [Uncomment if required] Generate training and test sets:
    # for p1 in (path.join(p,'train'),path.join(p,'test')):
    #     if path.exists(p1):
    #         shutil.rmtree(p1)
    # generate_train_test_sets(p)
    
    # [Experiment] Expand dataset (flips,rotations,etc.)
    # augment_dataset(opt.train_data_dir,'./data/train_aug')
    
    # [Experiment] Upsample with bicubic interpolation
    # generate_resized_dataset('./data/train_aug','./data/train_aug2')
