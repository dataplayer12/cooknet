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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

class ImageData(Dataset):
	def __init__(self, images, labels, size=[320, 240]):
		"""
		images: a list of N paths for images in training set
		labels: labels for images as list of length N
		"""
		
		assert len(images)==len(labels), 'images and labels not equal length'
		
		super(ImageData, self).__init__()
		self.image_paths=images
		self.labels=labels
		self.inputsize=size
		self.transforms=self.random_transforms()
		if self.labels is not None:
			assert len(self.image_paths)==self.labels.shape[0]
			#number of images and soft targets should be the same

	def random_transforms(self):
		normalize_transform=T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		#define normalization transform with which the torchvision models
		#were trained

		affine=T.RandomAffine(degrees=5, translate=(0.05, 0.05))
		hflip =T.RandomHorizontalFlip(p=0.7)
		vflip =T.RandomVerticalFlip(p=0.7)
		
		blur=T.GaussianBlur(7) #kernel size 5x5

		rt1=T.Compose([T.Resize(self.inputsize), affine, T.ToTensor(), normalize_transform])
		rt2=T.Compose([T.Resize(self.inputsize), hflip, T.ToTensor(), normalize_transform])
		rt3=T.Compose([T.Resize(self.inputsize), vflip, T.ToTensor(), normalize_transform])
		rt4=T.Compose([T.Resize(self.inputsize), blur, T.ToTensor(), normalize_transform])

		return [rt1, rt2, rt3, rt4]

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, index):
		imgpath=self.image_paths[index]
		img=Image.open(imgpath).convert('RGB') 
		#some images are grayscale and need to be converted into RGB
	
		img_tensor=self.transforms[torch.randint(0,4,[1,1]).item()](img)

		if self.labels is None:
			return img_tensor
		else:
			label_tensor=torch.tensor(self.labels[index,:])
			return img_tensor, label_tensor
			
			
class ClassificationManager(object):
	def __init__(self, indir, bsize=128):
		self.indir=indir
		self.nclasses, self.classes=self.findclasses()
		self.images, self.labels=self.getdata(shuffle=True)
		self.timages, self.tlabels, self.vimages, self.vlabels=self.split_train_valid(ratio=0.8)

		self.batchsize=bsize
		
		self.train_loader=self.get_train_loader()
		self.valid_loader=self.get_valid_loader()
		
		
	def getdata(self, shuffle=False):
		imgs=[]
		labels=[]
		for idx,cls in enumerate(self.classes):
			clsimgs=os.listdir(self.indir, cls)
			
			clsimgs=[f for f in clsimgs if f.endswith('.png')]
			imgs.extend([os.path.join(self.indir,cls,f) for f in clsimgs])
			labels.extend([idx]*len(clsimgs))
		
		if shuffle:
			order=np.random.permute(len(imgs))
			imgs=imgs[order]
			labels=labels[order]
		
		return imgs, labels
		
	def split_train_valid(self, ratio=0.8):
		split_index=int(ratio*len(self.images))
		#assume that data is already shuffled
		
		timgs=self.images[:split_index]
		tlbls=self.labels[:split_index]
		
		vimgs=self.images[split_index:]
		vlbls=self.labels[split_index:]
		
		return timgs,tlbls,vimgs,vlbls
		
		
	def findclasses(self):
		allfiles=sorted(os.listdir(self.indir))
		
		allclasses=[f for f in allfiles if os.path.isdir(os.path.join(self.indir,f))]
		
		return len(allclasses), allclasses
		
	def get_train_loader():
		pass
		tdata=ImageData(self.timages, self.tlabels)
		tloader=DataLoader(tdata, self.batchsize, shuffle=True, num_workers=8)
		
		return tloader
		
		
	def get_valid_loader():
		pass
		vdata=ImageData(self.vimages, self.vlabels)
		vloader=DataLoader(tdata, self.batchsize, shuffle=True, num_workers=8)
		return vloader
		
class CookNet(nn.Module):
	def __init__(self, nclasses, resnetpath):
		super(CookNet, self).__init__()
		self.nclasses=nclasses
		fullmodel=models.resnet18(pretrained=False)
		if os.path.exists(resnetpath):
			fullmodel.load_state_dict(torch.load(resnetpath, map_location=torch.device('cpu')))
		else:
			print('Could not find pretrained resnet at {}'.format(resnetpath))
			
		self.backbone=nn.Sequential(*list(fullmodel.children())[:-1])
		hidden_dim=list(fullmodel.children())[-1].in_features
		self.linear=nn.Linear(hidden_dim, self.nclasses)
		
	def forward(self, x):
		x=self.backbone(x)
		x=linear(x)
		
		return x
		
class CookTrainer(object):
	def __init__(self, net, dm):
		pass
		self.net=net
		self.dm=dm

		self.writer=SummaryWriter()

		self.optimizer=optim.Adam(self.net.parameters(), lr=1e-3)

	def train(self, epochs, save):
		pass
		eval_interval=1000

		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		train_loader, valid_loader = self.dm.train_loader, self.dm.valid_loader #ignore test loader if any

		self.net.to(device).train()

		self.net, self.optimizer = amp.initialize(self.net, self.optimizer,
									opt_level='O2', enabled=True)

		step=0
		
		get_accuracy=lambda p,y: (torch.argmax(p, dim=1) == y).to(torch.float).mean().item()

		for epoch in range(epochs):
			estart=time.time()
			for x,y in train_loader:
				self.optimizer.zero_grad()

				x=x.to(device)
				y=y.to(device)
				
				pred = self.net(x)
				
				loss = self.criterion(pred,y)

				self.writer.add_scalar('Loss', loss.item(), step)

				with amp.scale_loss(loss, self.optimizer) as scaled_loss:
					scaled_loss.backward()

				self.optimizer.step()
				acc=get_accuracy(pred, y)
				step+=1
				
				if step%eval_interval==0:
					self.student_network.eval()

					with torch.no_grad():
						pass
						for imgs, pred in valid_loader:
							imgs=imgs.to(device)
							stylized=self.student_network(imgs)
							self.writer.add_images('Stylized Examples', stylized, step)
							break #just one batch is enough

					self.student_network.train()

			self.save(epoch)
			eend=time.time()
			print('Time taken for last epoch = {:.3f}'.format(eend-estart))

	def save(self, epoch):
		if self.savepath:
			path=self.savepath.format(epoch)
			torch.save(self.student_network.state_dict(), path)
			print(f'Saved model to {path}')

		
def main():
	dm=ClassificationManager('./data')
	net=CookNet(insize=(240,320), outdims=dm.nclasses)
	trainer=CookNetTrainer(net,dm)
	trainer.train(epochs=100, save='cook_{}.pth')