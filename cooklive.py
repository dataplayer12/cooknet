from cooknet import CookNet

net=CookNet(2,None,'./models/oldcook_50.pth')

net.inferlive(save_path=None, really_not_steaming=True)
