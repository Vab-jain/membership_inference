import torch
from torch import nn
from pathlib import Path
import pickle
from config import device



def train_attack_models(config):
    # load attack model
    if config['attack_model'] == 'BasicNN':
        from attack_models import BasicNN
        attack_model = BasicNN(in_features=3).to(device)
    if config['attack_model'] == 'BasicNN_v2':
        from attack_models import BasicNN_v2
        attack_model = BasicNN_v2(in_features=3).to(device)

    # load attak dataset
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
    ctr_early_stop = 0
    prev_acc = 0
    new_acc = 0

    for epoch in range(config['train_attack_epochs']):
        print(f'Training Epoch: {epoch} / ' + str(config['train_attack_epochs']))
        attack_model.train()
        for batch_idx, (img, label) in enumerate(train_dataloader):
            optim.zero_grad()

            img = img.to(device)
            label = label.to(device)
            out = attack_model(img)

            loss =  loss_fn(out,label)
            loss.backward()

            optim.step()
            if not batch_idx%50:
                print(f'batch completed : {batch_idx}')

        print('Val begin')
        total = 0
        correct = 0
        attack_model.eval()
        with torch.no_grad():
            for _, (pos, label) in enumerate(val_dataloader):
                pos = pos.to(device)
                label = label.to(device)
                out = attack_model(pos)

                pred = torch.argmax(out, 1)
                
                total += label.size(0)
                correct += (pred == label).sum().item()
        
        new_acc = 100 * correct / total
        
        print(f'Val Accuracy of the model\nEpoch: {epoch}       Acc : {new_acc} %')
        
        if new_acc < prev_acc:
            ctr_early_stop += 1
        else:
            prev_acc = new_acc
            ctr_early_stop = 0
        
        if ctr_early_stop >= 10:
            print(f"EARLY STOPPING!!!!\nEpoch : {epoch}")
            break
        
    
    print('Training Finished')





    # SAVE MODEL
    SAVE_PATH = Path(str(Path.cwd()) + '/saved_attack_models/' + config['shadow_model'] + '_' + config['shadow_dataset'] + '.pth')

    torch.save(attack_model.state_dict(), SAVE_PATH)


if __name__ == '__main__':
    from config import config
    train_attack_models(config)