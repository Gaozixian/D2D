import torch
import torch.nn as nn
import torch.nn.functional as F
from LSTM.LSTM_predict import *
from model import *
""" 
-----加载config文件---
"""
config = load_config()


"""
--------这一步是dataset加载，加载控制信息序列，并且把对应的视觉图片对齐
"""


"""
--------这一步是编写model
"""