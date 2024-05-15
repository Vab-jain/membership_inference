import torch
from torch import nn
from attack_models import BasicNN
from pathlib import Path
import pickle
from config import device



def train_attack_models(config):
    # load shadow model
    attack_model = BasicNN(in_features=3).to(device)

    LOAD_PATH = Path(str(Path.cwd()) + '/attack_dataset/' + config['shadow_model'] + '_' + config['shadow_dataset'])

    with open(LOAD_PATH, "rb") as f:
        dataset = pickle.load(f)

    # split dataset into train and val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # load dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(attack_model.parameters(),lr=0.001,eps=1e-7)

    print('Training Started')

    for epoch in range(config['train_attack_epochs']):
        print(f'Training Epoch: {epoch} / ' + str(config['train_attack_epochs']))
        for batch_idx, (pos, label) in enumerate(train_dataloader):
            optim.zero_grad()

            pos = pos.to(device)
            out = attack_model(pos)

            loss =  loss_fn(out,label)
            # loss.requires_grad=True
            loss.backward()

            optim.step()
            if not batch_idx%50:
                print(f'batch completed : {batch_idx}')

    print('Training Finished')
    print('Val begin')
    total = 0
    correct = 0
    attack_model.eval()
    with torch.no_grad():
        for _, (pos, label) in enumerate(val_dataloader):
            pos = pos.to(device)
            out = attack_model(pos)

            _, pred = torch.max(out, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

    # SAVE MODEL
    SAVE_PATH = Path(str(Path.cwd()) + '/saved_attack_models/' + config['shadow_model'] + '_' + config['shadow_dataset'] + '.pth')

    torch.save(attack_model.state_dict(), SAVE_PATH)


if __name__ == '__main__':
    from config import config
    train_attack_models(config)