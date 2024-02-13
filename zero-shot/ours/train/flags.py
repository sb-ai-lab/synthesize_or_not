import argparse
import numpy as np

def parse_handle():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pics_series_path', help="path for saving series visualization", type=str, default='./1st_step_new_params_grad_clip/pics_series')
    parser.add_argument('--model_path', help="path for checkpoints", type=str, default='./1st_step_new_params_grad_clip/checkpoints')
    parser.add_argument('--pics_loss_path', help="path for saving loss and lr visualization", type=str, default='./1st_step_new_params_grad_clip/pics_loss')
    parser.add_argument('--lr', help="learning rate", type=str, default=3e-3)
    return parser