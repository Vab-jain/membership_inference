import torch
from torch import nn
import numpy as np
from pathlib import Path
import pickle
from torchvision import models
import typing
from config import device


def train_shadow_models(config):
# import the shadow dataset for training
# DATA_PATH = 'amlm/pickle/cifar10/resnet34/shadow.p'
    DATA_PATH = Path(str(Path.cwd())+'/datasets/'+ config['shadow_dataset'] + '/' + config['target_model'] + '/shadow.p')

    with open(DATA_PATH, "rb") as f:
        shadow_dataset = pickle.load(f)

    train_size = int(config['split_ratio'] * len(shadow_dataset))

    train_dataset = shadow_dataset[:train_size]

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

    shadow_model = models.resnet34(num_classes=10).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(shadow_model.parameters(),lr=0.001,eps=1e-7)

    print('Training Started')

    for epoch in range(config['train_shadow_epochs']):
        print(f'Training Epoch: {epoch} / ' + str(config['train_shadow_epochs']))
        for batch_idx, (img, label) in enumerate(dataloader):
            optim.zero_grad()

            img = img.to(device)
            out = shadow_model(img)

            loss =  loss_fn(torch.argmax(out, dim=1).float(),label.float())
            loss.requires_grad=True
            loss.backward()

            optim.step()
            if not batch_idx%50:
                print(f'batch completed : {batch_idx}')

    print('Training Finished')



    # SAVE MODEL
    SAVE_PATH = Path(str(Path.cwd()) + '/saved_shadow_models/' + config['shadow_model'] + '_' + config['shadow_dataset'] + '.pth')

    torch.save(shadow_model.state_dict(), SAVE_PATH)

