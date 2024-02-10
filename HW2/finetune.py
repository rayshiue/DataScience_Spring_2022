import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import transforms

def loss_fn_kd(outputs, labels, teacher_outputs, T, alpha):
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    return KD_loss

def FineTune(student, trainloader, dataset_sizes, teacher=None, lr = 0.01, T=4.5, alpha=0.1, simple = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sub_transform = transforms.Grayscale(num_output_channels=1)
    if simple:
        num_epochs = 20
        milestone = [10,15]
    else:
        num_epochs = 40
        milestone = [10,20,30]

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
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
                loss = loss_fn_kd(s_outputs, labels, t_outputs, T, alpha)
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item() * inputs.size(0)
            progress.refresh()
        progress.close()
        epoch_loss = running_loss / dataset_sizes
        print('Loss {:.4f}'.format(epoch_loss))
        if epoch in milestone:
            optimizer.param_groups[0]["lr"]*=0.1