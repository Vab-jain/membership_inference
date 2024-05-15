import torch

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


config = {
      'shadow_dataset'  : 'cifar10',
      'shadow_model'    : 'resnet34',
      'target_model'    : 'resnet34',
      'attack_model'    : 'BasicNN',
      'split_ratio'     : 0.5,
      'shadow_dataset_num_classes'    : 10,
      'train_attack_epochs'   : 2,
      'train_shadow_epochs'   : 2,
      'mode'            : 'eval',   # eval, test : (only for submission)
}
