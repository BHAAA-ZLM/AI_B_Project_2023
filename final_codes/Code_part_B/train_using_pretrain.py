import torch
import torchvision
import torch.nn as nn
from model import ManNatClassifier
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from torchvision import models
import os
import pandas as pd

torch.seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()


if __name__ == "__main__":
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    epoch_size = 10
    batchsize = 50
    lr = 3e-4
    print(f"Timestamp: {timestamp}, epoch_size: {epoch_size}, batchsize: {batchsize}, lr: {lr}, seed: {torch.seed}")
    loss_list = []
    accuracy_list = []
    train_set = torchvision.datasets.ImageFolder(
        root="./train",
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0
    )
    test_set = torchvision.datasets.ImageFolder(
        root="./test",
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(test_set, 
		batch_size=100,
		shuffle=False,
        num_workers=0)
    batch_num = len(train_set) // batchsize
    print(f"batch_num: {batch_num}, len(train_set): {len(train_set)}")
    test_data_iter = iter(test_loader)
    test_image, test_label = next(test_data_iter)
    models.resnet50()
    net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    net.fc = nn.Linear(512 * models.resnet.Bottleneck.expansion, 2)
    net.to(device)
    loss_function = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr=lr)  
    init = time.perf_counter() 
    for epoch in range(epoch_size): 
        running_loss = 0.0
        time_start = time.perf_counter()   
        for step, data in enumerate(train_loader, start=0):  
            inputs, labels = data 
            optimizer.zero_grad()  
            # forward + backward + optimize
            outputs = net(inputs.to(device))  	
            loss = loss_function(outputs, labels.to(device)) 
            loss.backward()
            optimizer.step() 
            running_loss += loss.item()
            if step % batch_num == batch_num - 1:
                with torch.no_grad(): 
                    net.eval()
                    outputs = net(test_image.to(device)) 
                    net.train()
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                        (epoch + 1, step + 1, running_loss / batch_num, accuracy))
                    loss_list.append(running_loss / batch_num)
                    accuracy_list.append(accuracy)
                    print('%f s' % (time.perf_counter() - time_start))
                    running_loss = 0.0
    print('Finished Training')
    print(f"Training time: {time.perf_counter() - init}")
    os.makedirs("./result", exist_ok=True)
    plt.plot(range(1, epoch_size + 1), loss_list, label="train_loss")
    plt.savefig(f"./result/{timestamp}-train-loss.png")
    plt.cla()
    plt.plot(range(1, epoch_size + 1), accuracy_list, label="eval_accuracy")
    plt.savefig(f"./result/{timestamp}-eval-acc.png")
    pd.DataFrame({"epoch":range(1, epoch_size + 1), "train_loss":loss_list, "accuracy_list":accuracy_list}).to_csv(f"./result/{timestamp}-train-loss.csv")

    save_path = f'./parameter/pretrain-{timestamp}.pth'
    print(f"Parameter save path: {save_path}")
    torch.save(net.state_dict(), save_path)
    