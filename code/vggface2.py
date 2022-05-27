import torch
import os, shutil
import glob
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# GPU 사용 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'{device} is available.')

# 경로 설정
train_path = '../../../data/islabdata/train'
test_path ='../../../data/islabdata/test'

def get_files_count(folder_path):
        dirListing = os.listdir(folder_path)
        return len(dirListing)

#print(get_files_count(train_path + '/real'))
#print(get_files_count(train_path + '/fake'))
#print(get_files_count(test_path + '/real'))
#print(get_files_count(test_path + '/fake'))

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = datasets.ImageFolder(root=train_path, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = datasets.ImageFolder(root=test_path, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

#VGGFace2 model
vggface2 = InceptionResnetV1(pretrained='vggface2', num_classes=2).to(device)
vggface2.last_linear = nn.Linear(in_features=1792, out_features=2).to(device)
vggface2.last_bn = nn.BatchNorm1d(2, eps=0.001, momentum=0.1, affine=True, track_running_stats=True).to(device)
vggface2.logits = nn.Linear(in_features=2, out_features=8631, bias=True).to(device)
vggface2 = nn.DataParallel(vggface2) #To allocate gpu

criterion = nn.CrossEntropyLoss().to(device)
learning_rate = 0.0001
epochs = 20
optimizer = optim.Adam(vggface2.parameters(), lr=learning_rate)

#Training
loss_ = []
n = len(trainloader)
print('Start Training')

for epoch in tqdm(range(epochs)):

    total_loss = 0.0
    total_accuracy = 0.0
    y_true = []
    y_pred = []
    for i, (images, labels) in enumerate(trainloader):
        images, labels= images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = vggface2(images)
        logits = F.softmax(outputs, dim=1)
    
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()    

        total_loss += loss.item()
        cost = loss.item()
        
        if i % 100 == 0:
            print('Epoch:' + str(epoch) + ", Iteration: " + str(i)
                + ", training cost = " + str(cost))
        
        # 정확도 계산
        pred = [torch.argmax(logit).cpu().detach().item() for logit in logits] 
        true = [label for label in labels.cpu().numpy()] 
        accuracy = accuracy_score(true, pred) 
        total_accuracy += accuracy

        y_pred.extend(pred)
        y_true.extend(true)

    avg_train_loss = total_loss / n
    avg_accuracy = total_accuracy / n

    print(f" {epoch+1} Epoch Average train loss : {avg_train_loss}") 
    print(f" {epoch+1} Epoch Average train accuracy : {avg_accuracy}")

print('Finished Training')


print('Classification Report (VGGFace2-Training):')
print(classification_report(y_true, y_pred, labels=[0,1], digits=4))

cm = confusion_matrix(y_true, y_pred, labels=[0,1])
ax = plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix - Training')

ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

ax.xaxis.set_ticklabels(['fake', 'real'])
ax.yaxis.set_ticklabels(['fake', 'real'])

fig = ax.get_figure()
fig.savefig("VGGFace2_train.png")

#Test
print("Start Testing")

t_pred = []
t_true = []

with torch.no_grad():

  vggface2.eval()
  for i, (images, labels) in enumerate(testloader):
    test_loss = 0.0
    test_accuracy = 0.0
    
    images, labels= images.to(device), labels.to(device)

    outputs = vggface2(images)
    logits = F.softmax(outputs, dim=1)
    
    loss = criterion(logits,labels)
    test_loss += loss.item()
    
    pred = [torch.argmax(logit).cpu().detach().item() for logit in logits] 
    true = [label for label in labels.cpu().numpy()] 
        
    accuracy = accuracy_score(true, pred) 
    test_accuracy += accuracy

    t_pred.extend(pred)
    t_true.extend(true)


avg_test_loss = test_loss / len(testloader) 
print(f"Test AVG loss : {avg_test_loss}") 
avg_test_accuracy = test_accuracy/len(testloader) 
print(f"Test AVG accuracy : {avg_test_accuracy: .2f}")



print('Classification Report (VGGFace2-Training):')
print(classification_report(t_true, t_pred, labels=[0,1], digits=4))

cm = confusion_matrix(t_true, t_pred, labels=[0,1])
ax = plt.subplot()

sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

ax.set_title('Confusion Matrix - Test')

ax.set_xlabel('Predicted Labels')
ax.set_ylabel('True Labels')

ax.xaxis.set_ticklabels(['fake', 'real'])
ax.yaxis.set_ticklabels(['fake', 'real'])
fig = ax.get_figure()
fig.savefig("VGGFace2_test.png")
