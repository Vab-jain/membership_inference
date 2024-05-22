import torch

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_config(task):
      if task=='task0':
            config = {
                  'task'            : 'task0',
                  'shadow_dataset'  : 'cifar10',
                  'shadow_model'    : 'resnet34',
                  'target_model'    : 'resnet34',
                  'attack_model'    : 'BasicNN',
                  'split_ratio'     : 0.5,
                  'shadow_dataset_num_classes'    : 10,
                  'train_attack_epochs'   : 50,
                  'train_shadow_epochs'   : 50,
                  'mode'            : 'eval',   # eval, test : (only for submission)
            }
      if task=='task1':
            config = {
                  'task'            : 'task1',
                  'shadow_dataset'  : 'cifar10',
                  'shadow_model'    : 'mobilenetv2',
                  'target_model'    : 'mobilenetv2',
                  'attack_model'    : 'BasicNN',
                  'split_ratio'     : 0.5,
                  'shadow_dataset_num_classes'    : 10,
                  'train_attack_epochs'   : 50,
                  'train_shadow_epochs'   : 50,
                  'mode'            : 'eval',   # eval, test : (only for submission)
            }
      if task=='task2':
            config = {
                  'task'            : 'task2',
                  'shadow_dataset'  : 'tinyimagenet',
                  'shadow_model'    : 'resnet34',
                  'target_model'    : 'resnet34',
                  'attack_model'    : 'BasicNN',
                  'split_ratio'     : 0.5,
                  'shadow_dataset_num_classes'    : 10,
                  'train_attack_epochs'   : 50,
                  'train_shadow_epochs'   : 50,
                  'mode'            : 'eval',   # eval, test : (only for submission)
            }
      if task=='task3':
            config = {
                  'task'            : 'task3',
                  'shadow_dataset'  : 'tinyimagenet',
                  'shadow_model'    : 'mobilenetv2',
                  'target_model'    : 'mobilenetv2',
                  'attack_model'    : 'BasicNN_v2',
                  'split_ratio'     : 0.5,
                  'shadow_dataset_num_classes'    : 10,
                  'train_attack_epochs'   : 50,
                  'train_shadow_epochs'   : 50,
                  'mode'            : 'eval',   # eval, test : (only for submission)
            }

      return config