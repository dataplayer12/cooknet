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

try:
	from RPi import GPIO
	GPIO.setmode(GPIO.BOARD)
	output_pin=18
	GPIO.setup(output_pin, GPIO.OUT)
	GPIO.output(output_pin,GPIO.HIGH)
	print('GPIO set up successfully and ready to control')
except:
	print('GPIO not found. Don\'t control')

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import os
import pdb
import cv2
import sys
from data.selectroi import getcoordinates

class ImageData(Dataset):
	def __init__(self, images, labels, nclasses,size=[320, 240]):
		"""
		images: a list of N paths for images in training set
		labels: labels for images as list of length N
		"""
		
		assert len(images)==len(labels), 'images and labels not equal length'
		
		super(ImageData, self).__init__()
		self.image_paths=images
		self.labels=labels
		self.nclasses=nclasses
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
			label_0d=torch.tensor(self.labels[index])
			label_1hot=nn.functional.one_hot(label_0d, num_classes=self.nclasses)
			eps=torch.rand(1)*0.1
			signs=torch.tensor([1,-1])
			noise=signs[label_1hot]*eps
			noisy_label=label_1hot+noise
			return img_tensor, noisy_label
			
			
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
		tdata=ImageData(self.timages, self.tlabels, self.nclasses)
		tloader=DataLoader(tdata, self.batchsize, shuffle=True, num_workers=8)
		return tloader
		
	def get_valid_loader(self):
		pass
		vdata=ImageData(self.vimages, self.vlabels, self.nclasses)
		vloader=DataLoader(vdata, self.batchsize, shuffle=True, num_workers=8)
		return vloader
		
class CookNet(nn.Module):
	def __init__(self, nclasses, resnetpath=None, loadpath=None):
		super(CookNet, self).__init__()
		self.nclasses=nclasses
		fullmodel=models.resnet18(pretrained=True)
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
			print('Loaded pretrained model from {}'.format(loadpath))
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
		#x=max(0,x-2)
		#y=max(0,y-2)
		#w+=4
		#h+=4
		cropped=frame[y:y+h,x:x+w,:]
		intensor=self.intt(cropped)[None,...].cuda()#.half()
		out=self.forward(intensor)
		out=nn.functional.softmax(out, dim=1).to('cpu').detach().numpy().astype(np.float32)
		#print(out.shape)
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

	def inferlive(self, cam='/dev/video0', 
		temp_path='data/0930_t.png', 
		save_path=None, 
		really_not_steaming=False, 
		cropdims=None,
		save_raw=False,
		do_control=False,
		get_cook_signal=False,
		display=True):

		src=cv2.VideoCapture(cam)
		time.sleep(1)
		fcount=0
		ret,frame=src.read()
		oldframe=frame[:]
		fps=0.0
		sprobs=[]
		if not ret:
			print('Cannot read camera')
			quit()

		with open('continue.txt','w') as f:
			f.write('Delete this file to stop cooking')

		dsti=None #handle for inferred video
		dstr=None #handle for raw video, used if save_raw=True
		spsave=None
		stop_condition=False #bool for condition to stop cooking,
		#used if do_control=True
		has_stopped=False

		if save_path:
			fps=src.get(cv2.CAP_PROP_FPS)
			h, w, _ = frame.shape
			if save_raw:
				raw_path=save_path[:save_path.rfind('.')]+'_r'+save_path[save_path.rfind('.'):]
				dstr=cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
				#|^| destination for raw video
			else:
				dstr=None

			inf_path=save_path[:save_path.rfind('.')]+'_i'+save_path[save_path.rfind('.'):]
			dsti=cv2.VideoWriter(inf_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
			#|^| destination for inferred video

			spsave=save_path[:save_path.rfind('.')]+'_probs.txt'

		self.eval()
		self.cuda()
		
		temp=cv2.imread(temp_path, 1)
		if cropdims is None:
			minx, miny, maxx, maxy = getcoordinates(frame, temp, exportlog=False)
			cropdims=[minx, miny, maxx-minx, maxy-miny]

		while ret:
			oldframe=frame[:]
			fcount+=1

			if dstr:
				dstr.write(oldframe)

			start=time.time()
			out, newframe=self.inferframe(frame, cropdims, True, fps)
			sprobs.append(out[0,1])
			endt=time.time()
			fps=0.9*fps+0.1/(endt-start)

			if dsti:
				dsti.write(newframe)

			if display:
				cv2.imshow('result', np.array(newframe))
				k=cv2.waitKey(1)
				if k==ord('q'):
					break

			if really_not_steaming:
				if out[0,1]>=0.8: #not steaming but still predicted as steam
					fname='./data/nosteam/{}.png'.format(int(1000*time.time()))
					cropped=oldframe[miny:maxy,minx:maxx,:]
					cv2.imwrite(fname, cropped)

			if do_control:
				if len(sprobs)%100==0:
					#smoothed_probability=np.mean(sprobs[-100:])
					stop=self.is_cooked(sprobs)
					if not has_stopped and stop:
						GPIO.output(output_pin, GPIO.LOW)
						has_stopped=time.time()
						print('STOP: Turned off IH')
						#IH will not turn on once switched off
					elif has_stopped:
						print('STOP: Previously turned off IH')
					else:
						print('RUN: Not cooked, continuing')

			ret,frame = src.read()
			
			if fcount%100==0 and not os.path.exists('continue.txt'):
				ret=False
				print('Received stop signal from file')

			if has_stopped:
				if (time.time()-has_stopped)>180: #3 minutes
					print('Finished cooking and cooled off. Goodbye')
					break

		if spsave:
			with open(spsave,'w') as f:
				f.write(str(sprobs))

		src.release()

		if dsti:
			dsti.release()
		if dstr:
			dstr.release()

		if get_cook_signal:
			signals=[]
			for x in range(0, len(sprobs), 100):
				pslice=sprobs[:x]
				_signal=self.is_cooked(pslice)
				signals.append(_signal)

			return signals

	def is_cooked(self, 
		probs, 
		chunksize=100, 
		pthreshold=0.5,
		mthreshold=0.2,
		sthreshold=0.2
		):
		'''
		probs: list of steam probability
		pthreshold: probability threshold
		mthreshold: max diff threshold in last 5 chunks
		sthreshold: sum threshold in last 5 chunks
		'''
		nchunks=5

		if len(probs)>=nchunks*chunksize:
			last5=np.array(probs[-nchunks*chunksize:]).reshape((nchunks,chunksize)).mean(axis=1)

			if last5[-1]>pthreshold:
				print('STOP: Last {} frames steam prob= {:.3f}'.format(chunksize, last5[-1]))
				return True

			mdiff=np.diff(last5).max()
			sdiff=np.diff(last5).sum() #this will also include negative values

			if mdiff>=mthreshold:
				print('STOP: Max change in last 5 chunks={:.3f}'.format(mdiff))
				return True
			if sdiff>=sthreshold:
				print('STOP: Sum of diffs in last 5 chunks={:.3f}'.format(sdiff))
				return True

			print('CONTINUE: Last {} frames steam prob= {:.3f}'.format(chunksize, last5[-1]))

		return False

class CrossEntropyWithLogitsLoss(nn.Module):
	def __init__(self):
		super(CrossEntropyWithLogitsLoss, self).__init__()

	def forward(self, model_output, noisy_targets):
		"""
		Takes in unnormalized logits from model and
		normalized but noisy target probabilities
		i.e. targets are not 0/1 but e/1-e, where e is small
		"""

		model_probabilities=nn.functional.softmax(model_output, dim=1)

		loss= - torch.mul(noisy_targets,torch.log(model_probabilities))

		return torch.mean(loss)

class CookTrainer(object):
	def __init__(self, net, dm):
		pass
		self.net=net
		self.dm=dm

		self.writer=SummaryWriter()
		self.criterion=CrossEntropyWithLogitsLoss() #nn.CrossEntropyLoss()
		self.optimizer=optim.AdamW(self.net.parameters(), lr=1e-6)
		self.savepath=None

	def train(self, epochs, save):
		pass
		eval_interval=200
		self.savepath=save
		device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		train_loader, valid_loader = self.dm.train_loader, self.dm.valid_loader #ignore test loader if any

		self.net.to(device).train()

		self.net, self.optimizer = amp.initialize(self.net, self.optimizer,
									opt_level='O2', enabled=True)

		step=0
		
		get_accuracy=lambda p,y: (torch.argmax(p, dim=1) == torch.argmax(y, dim=1)).to(torch.float).mean().item()

		for epoch in range(epochs):
			estart=time.time()
			for x,y in train_loader:
				self.optimizer.zero_grad()

				x=x.to(device)
				y=y.to(device)
				#pdb.set_trace()
				pred = self.net(x)
				#print(pred.shape, y.shape)
				
				loss = self.criterion(pred,y)

				#print(loss.item())
				self.writer.add_scalar('Training Loss', loss.item(), step)

				with amp.scale_loss(loss, self.optimizer) as scaled_loss:
					scaled_loss.backward()

				#loss.backward()

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
	dm=ClassificationManager('./data') #
	net=CookNet(nclasses=dm.nclasses, resnetpath='./resnet18_320p.pth', loadpath=None)
	trainer=CookTrainer(net,dm)
	trainer.train(epochs=100, save='models/cook_{}.pth')

if __name__=='__main__':
	main()
