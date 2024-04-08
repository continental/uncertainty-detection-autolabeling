# Copyright (C) 2024 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Moussa Kassem Sbeyti
# ==============================================================================
""" Runs the training script with a given config """


import subprocess
import configparser
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Read config file
config = configparser.ConfigParser()
config.read(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/configs/train/train_runner.ini')

# Read values from the config file
train_file_pattern = config['Paths']['train_file_pattern']
val_file_pattern = config['Paths']['val_file_pattern']
model_name = config['Paths']['model_name']
model_dir = config['Paths']['model_dir']
batch_size = config['Hyperparameters']['batch_size']
eval_samples = config['Hyperparameters']['eval_samples']
num_epochs = config['Hyperparameters']['num_epochs']
num_examples_per_epoch = config['Hyperparameters']['num_examples_per_epoch']
pretrained_ckpt = config['Hyperparameters']['pretrained_ckpt']
hparams = config['Hyperparameters']['hparams']

script_folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_folder)
# Construct the command
command = f"nohup python -m train_flags --train_file_pattern={train_file_pattern} --val_file_pattern={val_file_pattern} --model_name={model_name} --model_dir={model_dir} --batch_size={batch_size} --eval_samples={eval_samples} --num_epochs={num_epochs} --num_examples_per_epoch={num_examples_per_epoch} --pretrained_ckpt={pretrained_ckpt} --hparams={hparams} >> ./train.out"

# Execute the command
subprocess.run(command, shell=True)