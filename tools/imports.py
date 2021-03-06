import os
import sys
import numpy as np
import tqdm.notebook as tq
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as f
import torchvision
from torchvision.utils import save_image
from collections import OrderedDict
import threading
import queue
from math import pow
import skimage.transform
from pydensecrf import densecrf
import json


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__