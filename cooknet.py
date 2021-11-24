import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch
import numpy as np
import time

try:
	from torch.utils.tensorboard import SummaryWriter
	from apex import amp
except:
	print('tenosrboard and/or apex not available. Don\'t train')

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import os
import pdb
import cv2
import sys
from data.selectroi import getcoordinates

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
			assert len(self.image_paths)==len(self.labels)
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
			label_tensor=torch.tensor(self.labels[index])
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
			clsimgs=os.listdir(os.path.join(self.indir, cls))
			
			clsimgs=[f for f in clsimgs if f.endswith('.png')]
			imgs.extend([os.path.join(self.indir,cls,f) for f in clsimgs])
			labels.extend([idx]*len(clsimgs))
		
		if shuffle:
			order=np.random.permutation(len(imgs))
			imgs=[imgs[ox] for ox in order]
			labels=[labels[ox] for ox in order]
		
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
		allclasses=[f for f in allclasses if not f.startswith('.')]
		allclasses=[f for f in allclasses if not f.startswith('_')]
		print('Found {} classes: {}'.format(len(allclasses), allclasses))
		
		with open('labels.txt','w') as f:
			f.write('\n'.join(allclasses))

		return len(allclasses), allclasses
		
	def get_train_loader(self):
		pass
		tdata=ImageData(self.timages, self.tlabels)
		tloader=DataLoader(tdata, self.batchsize, shuffle=True, num_workers=8)
		return tloader
		
	def get_valid_loader(self):
		pass
		vdata=ImageData(self.vimages, self.vlabels)
		vloader=DataLoader(vdata, self.batchsize, shuffle=True, num_workers=8)
		return vloader
		
class CookNet(nn.Module):
	def __init__(self, nclasses, resnetpath=None, loadpath=None):
		super(CookNet, self).__init__()
		self.nclasses=nclasses
		fullmodel=models.resnet18(pretrained=False)
		if resnetpath and os.path.exists(resnetpath):
			fullmodel.load_state_dict(torch.load(resnetpath, map_location=torch.device('cpu')))
		else:
			print('Could not find pretrained resnet at {}'.format(resnetpath))
			
		self.backbone=nn.Sequential(*list(fullmodel.children())[:-1])
		self.flatten=nn.Flatten()
		hidden_dim=list(fullmodel.children())[-1].in_features
		self.linear=nn.Linear(hidden_dim, self.nclasses)

		if loadpath and os.path.exists(loadpath):
			self.load_state_dict(torch.load(loadpath, map_location=torch.device('cpu')))
		
		self.intt=T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		#pre-processing for input image

		if os.path.exists('labels.txt'):
			with open('labels.txt','r') as f:
				self.labels=f.read().split('\n')
		else:
			self.labels=['no steam','steam!']

	def forward(self, x):
		#print(x.shape)
		x=self.backbone(x)
		#print(x.shape)
		x=self.flatten(x)
		x=self.linear(x)
		
		return x

	def inferframe(self, frame, cropdims, annotate=True, fps=None):
		pass
		interpc=lambda x: (0,255-int(255*x),int(255*x)) #(0,255,0)-->(0,0,255)
		x,y,w,h=cropdims
		x=max(0,x-2)
		y=max(0,y-2)
		w+=4
		h+=4
		cropped=frame[y:y+h,x:x+w,:]
		intensor=self.intt(cropped)[None,...].cuda()
		out=self.forward(intensor)
		out=nn.functional.softmax(out).to('cpu').detach().numpy()
		color=interpc(out[0,1])
		clabel=self.labels[out[0].argmax()]

		if annotate:
			newframe=cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
			#newframe=cv2.putText(newframe, clabel, (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)
			if fps is not None:
				fpstr='FPS= {:.2f}'.format(fps)
				newframe=cv2.putText(newframe, fpstr, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color)
			return out, newframe

		return out

	def infervideo(self, infile, cropdims):
		src=cv2.VideoCapture(infile)
		ret,frame=src.read()
		

		if not ret:
			print('Could not read {}'.format(infile))
			quit()

		fps=float(src.get(cv2.CAP_PROP_FPS))
		w=int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
		h=int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
		#fcc=int(src.get(cv2.CAP_PROP_FOURCC))
		outname=infile[:infile.rfind('.')]+'_infer.mp4'

		if os.path.exists(outname):
			os.remove(outname)

		dst=cv2.VideoWriter(outname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
		print('Writing to {}'.format(outname))

		self.eval()

		sprobs=[]
		self.cuda()
		nframes=int(src.get(cv2.CAP_PROP_FRAME_COUNT))
		count=0
		with torch.no_grad():
			while ret:
				out, newframe = self.inferframe(frame, cropdims, annotate=True)
				sprobs.append(out[0,1])
				dst.write(newframe)
				ret,frame=src.read()
				count+=1
				sys.stdout.write("\rProgress: {}/{}".format(count,nframes))

		src.release()
		dst.release()
		with open('steamprobs.txt','w') as f:
			f.write(str(sprobs))

	def inferlive(self, cam='/dev/video0', temp_path='data/0930_t.png', save_path=None, really_not_steaming=False):
		src=cv2.VideoCapture(cam)
		time.sleep(1)
		ret,frame=src.read()
		oldframe=frame[:]
		fps=0.0
		if not ret:
			print('Cannot read camera')
			quit()

		dst=None

		if save_path:
			fps=src.get(cv2.CAP_PROP_FPS)
			h, w, _ = frame.shape
			dst=cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

		self.eval()
		self.cuda()
		temp=cv2.imread(temp_path, 1)
		minx, miny, maxx, maxy = getcoordinates(frame, temp, exportlog=True)
		cropdims=[minx, miny, maxx-minx, maxy-miny]
		while ret:
			start=time.time()
			out, newframe=self.inferframe(frame, cropdims, True, fps)
			endt=time.time()
			fps=0.9*fps+0.1/(endt-start)

			if dst:
				dst.write(newframe)
			cv2.imshow('result', np.array(newframe))
			k=cv2.waitKey(1)
			if k==ord('q'):
				break

			if really_not_steaming:
				if out[0,1]>=0.8: #not steaming but still predicted as steam
					fname='./data/nosteam/{}.png'.format(int(1000*time.time()))
					cropped=oldframe[miny:maxy,minx:maxx,:]
					cv2.imwrite(fname, cropped)

			ret,frame = src.read()
			oldframe=frame[:]
		src.release()
		dst.release()


class CookTrainer(object):
	def __init__(self, net, dm):
		pass
		self.net=net
		self.dm=dm

		self.writer=SummaryWriter()
		self.criterion=nn.CrossEntropyLoss()
		self.optimizer=optim.Adam(self.net.parameters(), lr=1e-5)
		self.savepath=None

	def train(self, epochs, save):
		pass
		eval_interval=200
		self.savepath=save
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		train_loader, valid_loader = self.dm.train_loader, self.dm.valid_loader #ignore test loader if any

		self.net.to(device).train()

		self.net, self.optimizer = amp.initialize(self.net, self.optimizer,
									opt_level='O0', enabled=False)

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

				self.writer.add_scalar('Training Loss', loss.item(), step)

				with amp.scale_loss(loss, self.optimizer) as scaled_loss:
					scaled_loss.backward()

				torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)
				self.optimizer.step()
				acc=get_accuracy(pred, y)
				step+=1
				self.writer.add_scalar('Training Accuracy', acc, step)
				#print('Accuracy={:.4f}'.format(acc))

				if step%eval_interval==0:
					self.net.eval()
					valoss=[]
					vaacc=[]
					with torch.no_grad():
						pass
						for imgs, ys in valid_loader:
							imgs=imgs.to(device)
							ys=ys.to(device)
							preds=self.net(imgs)
							vacc=get_accuracy(preds, ys)
							vloss=self.criterion(preds, ys)
							#pdb.set_trace()
							valoss.append(vloss.flatten().item())
							vaacc.append(vacc)

					#print(valoss[0])
					self.writer.add_scalar('Validation Loss', np.mean(valoss), step)
					self.writer.add_scalar('Validation Accuracy', np.mean(vaacc), step)
					self.net.train()

			self.save(epoch)
			eend=time.time()
			print('Time taken for last epoch = {:.3f}'.format(eend-estart))

	def save(self, epoch):
		if self.savepath:
			path=self.savepath.format(epoch)
			torch.save(self.net.state_dict(), path)
			print(f'Saved model to {path}')

		
def main():
	dm=ClassificationManager('./data')
	net=CookNet(nclasses=dm.nclasses, resnetpath='./resnet18_320p.pth', loadpath=None)
	trainer=CookTrainer(net,dm)
	trainer.train(epochs=100, save='models/cook_{}.pth')

if __name__=='__main__':
	main()
