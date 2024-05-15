from config import config, device
from train_shadow_models import train_shadow_models
from create_attack_dataset import create_attack_dataset
from train_attack_models import train_attack_models
from membership_inference import membership_inference

'''
# get args from cmd
# args <= shadow_dataset_name, shadow_model, attack_model, [eval/test]_dataset

# parser = argparse.ArgumentParser()

# parser.add_argument('shadow_model', help='name of the shadow model architecture to use; supported models are resnet34, [add more models here...]')

'''

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