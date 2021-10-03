import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch
import numpy as np
from dataset import DataManager
import config as cfg
import time
from torch.utils.tensorboard import SummaryWriter
from apex import amp

class ClassificationManager(object):
	def __init__(self, indir):
		self.indir=indir
		

def main():
	dm=ClassificationManager('./data')
	net=CookNet(insize=(240,320), outdims=dm.nclasses)
	trainer=CookNetTrainer(net,dm)
	trainer.train(epochs=100, save='cook_{}.pth')