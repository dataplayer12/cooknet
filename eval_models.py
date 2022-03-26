from cooknet import CookNet
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

if len(sys.argv) !=3:
	print('Usage: python3 eval_models.py videopath templatepath')
	quit()

video=sys.argv[1]
temp_path=sys.argv[2]

models=['models/20220112_cook_99.pth', 
	'models/20220123_cook_99.pth',
	'models/oldcook_50.pth',
	'models/t-1-2-cook_21.pth',
	'models/cook_25.pth',
	'models/cook_50.pth',
	'models/cook_75.pth',
	'models/cook_99.pth'
	]

# models=['models/20220112_cook_99.pth', 
# 	'models/20220123_cook_99.pth',
# 	]

networks=[]

probabilities={}

for mname in models:
	nobj=CookNet(2,None,mname).eval()
	networks.append(nobj)


#net.inferlive(save_path='../livecook0127.mp4', really_not_steaming=False)
#net.inferlive(cam='../cooker0127.mp4', save_path='../infer0127_99_full.mp4') #, cropdims=[202,0,320,240])

for mname, net in zip(models, networks):
	mfile=mname[mname.rfind('/')+1:mname.rfind('.')]
	vfile=video[video.rfind('/')+1:]
	savepath='evals/'+mfile+vfile

	is_cooked=net.inferlive(
		cam=video,
		temp_path=temp_path, 
		really_not_steaming=False, #this is not a MLOps-ish run
		cropdims=None,
		save_path=savepath,
		save_raw=False,
		do_control=False,
		get_cook_signal=True,
		display=False
		)
	
	probabilities[mname]=is_cooked


fig=plt.figure()

for mname,vprobs in probabilities.items():
	mfile=mname[mname.rfind('/')+1:mfile.rfind('.')]
	if mfile.startswith('cook'):
		mlabel='epoch_'+mfile.split('_')[-1]
	else:
		mlabel=mfile

	plt.plot(vprobs, label=mlabel)


fig.legend(loc='upper left')
plt.xlabel('Frame #')
plt.ylabel('Cooked bool signal')
plt.show()

figname='eval_on_'+video[video.rfind('/')+1:video.rfind('.')]+'.png'
plt.savefig(figname)