import torch
from torch import nn
import numpy as np
from pathlib import Path
import pickle
from torchvision import models
import typing
from config import device, config


def train_shadow_models(config):
# import the shadow dataset for training
# DATA_PATH = 'amlm/pickle/cifar10/resnet34/shadow.p'
    DATA_PATH = Path(str(Path.cwd())+'/datasets/'+ config['shadow_dataset'] + '/' + config['target_model'] + '/shadow.p')

    with open(DATA_PATH, "rb") as f:
        shadow_dataset = pickle.load(f)

    train_size = int(config['split_ratio'] * len(shadow_dataset))

    train_dataset, val_dataset = shadow_dataset[:train_size-1000], shadow_dataset[train_size-1000:train_size]

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

    shadow_model = models.resnet34(num_classes=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(shadow_model.parameters(),lr=0.001,eps=1e-7)           # in paper lr=0.0001,

    print('Training Started')

    for epoch in range(config['train_shadow_epochs']):
        print(f'Training Epoch: {epoch} / ' + str(config['train_shadow_epochs']))
        for batch_idx, (img, label) in enumerate(train_dataloader):
            optim.zero_grad()

            img = img.to(device)
            out = shadow_model(img)

            # Apply softmax to get probabilities
            predictions = torch.softmax(out, dim=1)

            loss =  loss_fn(predictions.float(),label.long())
            # loss.requires_grad=True
            loss.backward()

            optim.step()
            if not batch_idx%50:
                print(f'batch completed : {batch_idx}')

    print('Training Finished')

    print('Val begin')
    total = 0
    correct = 0
    shadow_model.eval()
    with torch.no_grad():
        for _, (pos, label) in enumerate(val_dataloader):
            pos = pos.to(device)
            out = shadow_model(pos)

            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))



    # SAVE MODEL
    SAVE_PATH = Path(str(Path.cwd()) + '/saved_shadow_models/' + config['shadow_model'] + '_' + config['shadow_dataset'] + '.pth')

    torch.save(shadow_model.state_dict(), SAVE_PATH)



if __name__ == "__main__":
    train_shadow_models(config)