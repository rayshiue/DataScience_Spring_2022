import torch
from torch import nn
import torchvision
import torch.nn.functional as F
from special_resnet import ResNet50, SpecialResNet
from tqdm import tqdm
from torchsummary import summary
import torch_pruning as tp
import pandas as pd
import transforms
from finetune import FineTune
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_fn_kd(outputs, labels, teacher_outputs, T, alpha):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(probability = 0.5, sh = 0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]),
    ])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
dataset_sizes = len(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=0)
tran = torchvision.transforms.Grayscale()

checkpoint = torch.load('./resnet-50.pth')
teacher = ResNet50().to(device)
teacher.load_state_dict(checkpoint['model_state_dict'])
teacher.eval()

alpha = 0.1
temperature = 4
lr = 0.1
lr_decay_gamma = 0.1
num_epochs = 100
milestone = [40, 75]

student =SpecialResNet(depth=20, num_classes = 10).to(device)
summary(student,(1,28,28))

sub_transform = transforms.Grayscale(num_output_channels=1)
optimizer = torch.optim.SGD(student.parameters(),lr=lr, momentum=0.9,weight_decay=5e-4)


#Knowledge Distillation
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)
    running_loss = 0.0

    progress = tqdm(total=len(trainloader), position=0, leave=True)
    for inputs, labels in trainloader:
        progress.update(1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            s_outputs = student(sub_transform(inputs))
            t_outputs = teacher(inputs)
            loss = loss_fn_kd(s_outputs, labels, t_outputs, temperature, alpha)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        progress.refresh()
    progress.close()
    print(f'Loss: {running_loss / dataset_sizes:.4f}')

    if epoch in milestone:
        l = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[0]["lr"]*=lr_decay_gamma
        print('Learning rate changes from {} into {}'.format(l, optimizer.param_groups[0]["lr"]))

#Prune
example_inputs = torch.randn(1, 1, 28, 28)
example_inputs = example_inputs.to(device)
imp = tp.importance.GroupNormImportance(p=2)
ignored_layers = [student.fc]
iterative_steps = 5
pruner = tp.pruner.MagnitudePruner( 
    student,
    example_inputs,
    importance=imp,
    iterative_steps=iterative_steps,
    ch_sparsity=0.38,
    ignored_layers=ignored_layers,
)

print('=========================')
print('      Start Pruning      ')
print('=========================')
for i in range(iterative_steps):
    print('=========================')
    print('   Pruning Iteration {}  '.format(i+1))
    print('=========================')
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(student, example_inputs)
    print("Number of parameters : {}".format(nparams))
    FineTune(student, trainloader, dataset_sizes, teacher, lr = 0.0001, T=temperature, alpha=alpha, simple =True)
torch.save(student,'./pruned_model.pth')
#Predict
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

correct = 0
total = 0
pred_arr = []

progress = tqdm(total=len(testloader), position=0, leave=True)
with torch.no_grad():
    for data in testloader:
        progress.update(1)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = student(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pred_arr.append(predicted.item())
progress.close()
accuracy = 100 * correct / total
print(f"Accuracy of the network on the {total} test images: {accuracy:.2f} %")

pred_data = {"pred":pred_arr}
df_pred = pd.DataFrame(pred_data)
df_pred.to_csv('example_pred.csv', index_label='id')