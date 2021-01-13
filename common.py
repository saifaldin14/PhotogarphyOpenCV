import rawpy
import cv2
import numpy as np

def load_image(path, bps=16):
    #bps allows us to control the precision, dow e want the full 16 bits
    #or it 8 bits good enough
    if path.suffix == '.CR2':
        #If the file is in .CR2 (Cannon format) format we open and extract it with rawpy
        #without doing any post processing since we want to use openCV
        # for that
        with rawpy.imread(str(path)) as raw:
            data = raw.postprocess(no_auto_bright=True,
                                   gammma=(1,1),
                                   output_bps=bps)
            return data
    else:
        #For anything that is not .CR2 use openCV instead
        return cv2.imread(str(path))

def load_14bit_gray(path):
    img = load_image(path, bps=16)
    #Cannon uses RGB format whereas the default in openCV is BGR so we switch it
    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 4).astype(np.uint16)
