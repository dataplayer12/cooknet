from cooknet import CookNet
import sys
mpath= './models/20220326_cook_31.pth' #'./models/20220319_cook_99.pth'
net=CookNet(2,None,mpath)#.half()

#net.inferlive(save_path='../livecook0127.mp4', really_not_steaming=False)
#net.inferlive(cam='../cooker0127.mp4', save_path='../infer0127_99_full.mp4') #, cropdims=[202,0,320,240])

net.inferlive(
	save_path=sys.argv[1],
	temp_path= 'data/1031_round.png', #'data/op1002_t.png', #'data/0930_t.png', 
	really_not_steaming=False, #this is not a MLOps-ish run
	cropdims=None,
	save_raw=True,
	do_control=True
	)
