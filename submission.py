import torch
from pathlib import Path
from config import device
import pickle
import torchvision.models as models
import numpy as np

def submission_pipeline(config):
    print('{} MODE!!!'.format(config['mode']))
    
    LOAD_MODEL_PATH = Path(str(Path.cwd()) + '/saved_attack_models/' + config['shadow_model'] + '_' + config['shadow_dataset'] + '.pth')
    LOAD_DATA_PATH = Path(str(Path.cwd())+'/datasets/'+ config['shadow_dataset'] + '/' + config['target_model'] + '/' + config['mode'] + '.p')
    LOAD_TARGET_MODEL_PATH = Path(str(Path.cwd()) + '/target_models/' + config['target_model'] + '_' + config['shadow_dataset'] + '.pth')

    if config['attack_model'] == 'BasicNN':
        from attack_models import BasicNN
        attack_model = BasicNN(in_features=3).to(device)
    if config['attack_model'] == 'BasicNN_v2':
        from attack_models import BasicNN_v2
        attack_model = BasicNN_v2(in_features=3).to(device)
    attack_model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
    # add more models here in if-else ladder

    if config['shadow_dataset']=='cifar10':
        if config['target_model'] == 'resnet34':
            target_model = models.resnet34(num_classes=10).to(device)  
        if config['target_model'] == 'mobilenetv2':
            target_model = models.mobilenet_v2(num_classes=10).to(device)
    if config['shadow_dataset']=='tinyimagenet':
        if config['target_model'] == 'resnet34':
            target_model = models.resnet34(num_classes=200).to(device)  
        if config['target_model'] == 'mobilenetv2':
            target_model = models.mobilenet_v2(num_classes=200).to(device)
    state_dict = torch.load(LOAD_TARGET_MODEL_PATH, map_location=device)
    target_model.load_state_dict(state_dict['net'])

    with open(LOAD_DATA_PATH, "rb") as f:
        dataset = pickle.load(f)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    target_model.eval()
    attack_model.eval()
    
    submission = []
    
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device)
            
            pred = target_model(img)
            top_pred = torch.topk(pred, 3)[0]

            out = attack_model(top_pred)
            
            if config['mode']=='test':
                submission.extend(out.argmax(dim=1).tolist())
    
    np.save(f'{config["task"]}_{config["target_model"]}_{config["shadow_dataset"]}.npy', submission)


if __name__ == '__main__':
    from config import get_config, device
    submission_pipeline(get_config('task0'))