import cv2
import os


def getcoordinates(vname,tname, channel='red', exportlog=False):
    """
    vname: video name (string) or numpy ndarray video frame
    tname: img filename (string) or numpy ndarray
    """
    cmappings={'blue':0,'green':1,'red':2,'all':[0,1,2]}

    if type(vname)==str:
        src=cv2.VideoCapture(vname)
        fps=src.get(cv2.CAP_PROP_FPS)
        src.set(cv2.CAP_PROP_POS_FRAMES, int(60*fps))

        ret,frame=src.read()
        if not ret:
            print('Could not read video {}'.format(vname))
            quit()
    else:
        frame=vname[:] #vname can be just a numpy array

    if type(tname)==str:
        temp=cv2.imread(tname,1)
    else:
        temp=tname[:]

    temp=temp[:,:,cmappings[channel]]
    frame=frame[:,:,cmappings[channel]]

    if len(temp.shape)==3:
        temp=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY)
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tmatch=cv2.matchTemplate(frame,temp, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(tmatch)
    top_left = max_loc

    center_pt = (top_left[0] + temp.shape[1]//2, top_left[1] + temp.shape[0]//2)
    
    #bottom_right = (top_left[0] + temp.shape[1], top_left[1] + temp.shape[0])

    minx,miny = center_pt[0]-frame.shape[1]//4, center_pt[1]-frame.shape[0]//4
    maxx,maxy = center_pt[0]+frame.shape[1]//4, center_pt[1]+frame.shape[0]//4

    if minx<0:
        minx = 0
        maxx = frame.shape[1]//2

    if miny<0:
        miny = 0
        maxy = frame.shape[0]//2

    if maxx>=frame.shape[1]:
        maxx = frame.shape[1]
        minx = frame.shape[1]//2

    if maxy >= frame.shape[0]:
        maxy = frame.shape[0]
        miny = frame.shape[0]//2

    #src.release()
    
    if exportlog:
        if type(vname)==str:
            ret, frame=src.read()
        newframe=cv2.rectangle(frame, (minx,miny), (maxx, maxy), (255,0,255), 2)
        cv2.imwrite('demo_rect.png', newframe)

    if type(vname)==str:
        src.release()

    return [minx, miny, maxx, maxy]

def seconds2HMSstring(tstamp):
    hh = tstamp//60//60
    mm = tstamp//60 - 60*hh
    ss = tstamp%60

    hmstr='{:02d}:{:02d}:{:02d}'.format(hh,mm,ss)

    return hmstr

def generate_images(vname, tname, break_point, widths, channel='red'):
    
    if widths[1]==0:
        src=cv2.VideoCapture(vname)
        fps=src.get(cv2.CAP_PROP_FPS)
        fcount=src.get(cv2.CAP_PROP_FRAME_COUNT)
        nseconds=int(fcount/fps)
        src.release()
        widths[1]=nseconds-break_point

    minx, miny, maxx, maxy = getcoordinates(vname, tname, channel)
    #[233,73,553,313]
    stime = seconds2HMSstring(break_point - widths[0])
    etime = seconds2HMSstring(break_point)
    ftime = seconds2HMSstring(break_point+widths[1])

    vprefix = vname[:vname.rfind('.')][-4:]

    ffstr1=f'ffmpeg -i {vname} -ss {stime} -to {etime} -vf crop=iw/2:ih/2:{minx}:{miny} nosteam/{vprefix}_%d.png'

    ffstr2=f'ffmpeg -i {vname} -ss {etime} -to {ftime} -vf crop=iw/2:ih/2:{minx}:{miny} steam/{vprefix}_%d.png'

    print('Generating nosteam images...')
    os.system(ffstr1)

    print('----------')

    print('Generating steam images...')
    os.system(ffstr2)

if __name__=='__main__':
    pass
    # generate_images('cooker0930.mp4', '0930_round.png', 2645, [360, 0])
    # generate_images('openpan1002.mp4', 'op1002_t.png', 1350, [360, 0], channel='all')
    # generate_images('cook_0215.mp4', 'op1002_t.png', 140, [135, 0], channel='all')
    # generate_images('cook_0220.mp4', '1031_round.png', 1208, [360, 0])
    # generate_images('cook_0306.mp4', '1031_round.png', 985, [360, 0])
    # generate_images('cook_0313.mp4', '1031_round.png', 1220, [360, 0])
    generate_images('cook_0326.mp4', '1031_round.png', 648, [120, 0])
    # generate_images('cooker1013.mp4', '138.png', 3885, [600,0])
    # generate_images('cooker1031.mp4', '1031_round.png', 2840, [600, 0]) #202,0,522, 240
    #print(getcoordinates('cooker1107.mp4', '0930_t.png', exportlog=True))
    pass
