from PIL import Image
from setting import *

class LoadData():
    def __init__(self, path, dataset='Flickr-25K'):
        print ('******************************************************')
        print 		('dataset:{0}'.format(path))
        print ('******************************************************')

    def loadimg(self, pathList):
        crop_size = 224
        ImgSelect = np.ndarray([len(pathList), crop_size, crop_size, 3])
        count = 0
        for path in pathList:
            img = Image.open(path)
            xsize, ysize = img.size
            # ***************************************************************************************************
            # Here, we fist resize the original iamge into M*224 or 224*M, M>224, then cut out the part of M-224 surround.
            seldim = min(xsize, ysize)
            rate = 224.0 / seldim
            img = img.resize((int(xsize * rate), int(ysize * rate)))
            nxsize, nysize = img.size
            box = (nxsize / 2.0 - 112, nysize / 2.0 - 112, nxsize / 2.0 + 112, nysize / 2.0 + 112)
            img = img.crop(box)
            img = img.convert("RGB")
            img = img.resize((224, 224))
            img = np.array(img)
            if img.shape[2] != 3:
                print ('This image is not a rgb picture: {0}'.format(path))
                print ('The shape of this image is {0}'.format(img.shape))
                ImgSelect[count, :, :, :] = img[:, :, 0:3]
                count += 1
            else:
                ImgSelect[count, :, :, :] = img
                count += 1
        return ImgSelect
