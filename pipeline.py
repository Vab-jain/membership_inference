from config import get_config
from train_shadow_models import train_shadow_models
from create_attack_dataset import create_attack_dataset
from train_attack_models import train_attack_models
from membership_inference import membership_inference
from submission import submission_pipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='Set the program mode.', choices=['eval', 'test'], required=True)
parser.add_argument('-t', '--task', help='Specify the task to run. Choose from task0, task1, task2, or task3.', choices=['task0', 'task1', 'task2', 'task3'], required=True)

args = vars(parser.parse_args())

mode = args['mode']

config = get_config('task2')
config['mode'] = mode

if mode=='eval':
    # train shadow models
    # def train_shadow_models(model, dataset_path) -> /saved_shadow_models/{model}_{datset}.pth
    train_shadow_models(config)

    # create attack dataset
    # def create_attack_dataset(saved_shadow_model_path, train_dataset_path, test_dataset_path) -> attack_dataset_{shadow_model}.p
    create_attack_dataset(config)

    # train attack model
    # def train_attack_models(model, attack_dataset_path) -> /saved_attack_models/{model}_{dataset}.pth
    train_attack_models(config)

    # run membership attack
    # def membership_inference(inference_dataset_path, attack_model_path) -> task{i}_{dataset}.npy
    membership_inference(config)
    
if mode=='test':
    # run membership attack
    # def membership_inference(inference_dataset_path, attack_model_path) -> task{i}_{dataset}.npy
    submission_pipeline(config)