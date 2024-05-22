import torch
from torchvision import models
from pathlib import Path
import pickle
import numpy as np
from config import device


def create_attack_dataset(config):

    # load shadow model
    LOAD_PATH = Path(str(Path.cwd())+'/saved_shadow_models' + '/' + config['shadow_model'] + '_' + config['shadow_dataset'] + '.pth')

    if config['shadow_dataset']=='cifar10':
        if config['shadow_model'] == 'resnet34':
            shadow_model = models.resnet34(num_classes=10).to(device)
        if config['shadow_model'] == 'mobilenetv2':
            shadow_model = models.mobilenet_v2(num_classes=10).to(device)
    if config['shadow_dataset']=='tinyimagenet':
        if config['shadow_model'] == 'resnet34':
            shadow_model = models.resnet34(num_classes=200).to(device)
        if config['shadow_model'] == 'mobilenetv2':
            shadow_model = models.mobilenet_v2(num_classes=200).to(device)
    shadow_model.load_state_dict(torch.load(LOAD_PATH))

    # load shadow datasets 
    DATA_PATH = Path(str(Path.cwd())+'/datasets/'+ config['shadow_dataset'] + '/' + config['target_model'] + '/shadow.p')

    with open(DATA_PATH, "rb") as f:
        shadow_dataset = pickle.load(f)

    train_size = int(config['split_ratio'] * len(shadow_dataset))

    train_dataset = shadow_dataset[:train_size]
    # train_dataset = train_dataset[:20]
    test_dataset = shadow_dataset[train_size:]
    # test_dataset = test_dataset[:10]

    # load dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # in eval mode, get predictions for train/test data from the shadow model
    # and save the prediction into a tensor
    # pass both train and test data through shadow model
    # and label each sample with 'in': train  and  'out': test
    attack_dataset = []
    shadow_model.eval()
    with torch.no_grad():
        for _, (img, label) in enumerate(train_dataloader):
                img = img.to(device)
                preds = shadow_model(img)
                for pred in preds:
                    # pick only top3 predictions
                    top_pred = torch.topk(pred, 3)[0]
                    attack_dataset.append(([top_pred, torch.tensor(1)]))

        for _, (img, label) in enumerate(test_dataloader):
                img = img.to(device)
                preds = shadow_model(img)
                for pred in preds:
                    # pick only top3 predictions
                    top_pred = torch.topk(pred, 3)[0]
                    attack_dataset.append([top_pred, torch.tensor(0)])

    
    
    # SAVE ATTACK DATASET  
    SAVE_PATH = Path(str(Path.cwd()) + '/attack_dataset/' + config['shadow_model'] + '_' + config['shadow_dataset'])

    # torch.save(torch.stack(attack_dataset, dim=0), SAVE_PATH)
    with open(SAVE_PATH, 'w+b') as f:
        pickle.dump(attack_dataset, f)



if __name__ == '__main__':
    from config import config, device
    create_attack_dataset(config)