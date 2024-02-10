import transforms
import torch
import torchvision
from tqdm import tqdm
from torchsummary import summary
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

correct = 0
total = 0
pred_arr = []

student = torch.load('./pruned_model.pth')
summary(student, (1,28,28))

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