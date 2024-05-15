import torch
from pathlib import Path
from config import device
import pickle
import torchvision.models as models

def membership_inference(config):
    LOAD_MODEL_PATH = Path(str(Path.cwd()) + '/saved_attack_models/' + config['shadow_model'] + '_' + config['shadow_dataset'] + '.pth')
    LOAD_DATA_PATH = Path(str(Path.cwd())+'/datasets/'+ config['shadow_dataset'] + '/' + config['target_model'] + '/' + config['mode'] + '.p')
    LOAD_TARGET_MODEL_PATH = Path(str(Path.cwd()) + '/target_models/' + config['target_model'] + '_' + config['shadow_dataset'] + '.pth')

    if config['attack_model'] == 'BasicNN':
        from attack_models import BasicNN
        attack_model = BasicNN(in_features=3).to(device)
        attack_model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=device))
    # add more models here in if-else ladder

    if config['target_model'] == 'resnet34':
        target_model = models.resnet34(num_classes=10).to(device)
        state_dict = torch.load(LOAD_TARGET_MODEL_PATH, map_location=device)
        target_model.load_state_dict(state_dict['net'])


    with open(LOAD_DATA_PATH, "rb") as f:
        dataset = pickle.load(f)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    target_model.eval()
    attack_model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (img, label, membership) in enumerate(dataloader):
            img = img.to(device)
            pred = target_model(img)
            top_pred = torch.topk(pred, 3)[0]

            out = attack_model(top_pred)

            total += label.size(0)
            correct += (out.argmax(dim=1) == membership).sum().item()
    
    print('---> Results of the membership inference attack <---')
    print('Final accuracy on eval is : ' + str(100 * correct / total))


if __name__ == '__main__':
    from config import config, device
    membership_inference(config)