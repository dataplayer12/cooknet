from cooknet import CookNet
import sys

net=CookNet(2,None,'./models/20220123_cook_99.pth')#.half()

#net.inferlive(save_path='../livecook0127.mp4', really_not_steaming=False)
#net.inferlive(cam='../cooker0127.mp4', save_path='../infer0127_99_full.mp4') #, cropdims=[202,0,320,240])

net.inferlive(save_path=sys.argv[1],
	really_not_steaming=False, #this is not a MLOps-ish run
	cropdims=None,
	save_raw=True,
	do_control=True
	)
