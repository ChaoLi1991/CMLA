import numpy as np
import h5py
from load_mir import LoadData

select_gpu = '0'
per_process_gpu_memory_fraction = 0.7

Param = {}

#========================================Flickr-25K====================================#
Param['DATA_DIR']  = '/home/chao/HashWorks/Project/Flickr-25K'
Param['Imgpath']  = '/home/chao/HashWorks/Project/Flickr-25K/Img.h5'
Param['Tagpath']  = '/home/chao/HashWorks/Project/Flickr-25K/Tag.h5'
Param['Labpath']  = '/home/chao/HashWorks/Project/Flickr-25K/Lab.h5'
Param['Meanpath']  = '/home/chao/HashWorks/Project/Flickr-25K/Mean.h5'

Param['dataset_dir'] = 'Flickr-25K'
Param['MODEL_DIR'] = '/home/chao/HashWorks/model/imagenet-vgg-f.mat' #discriminator_img  pretrain model
Param['adv_dir'] = '/home/chao/HashWorks/NIPS19/AttackHash_SSAH_Flickr/AdversarialSamples/'
Param['noi_dir'] = '/home/chao/HashWorks/NIPS19/AttackHash_SSAH_Flickr/NoiseSamples/'
Param['checkpoint_dir'] = '/export/Data/CrossModalData/SaveData/Project/NIPS19/AttackHash_SSAH_Flickr/checkpoint'
Param['code_dir'] = '/export/Data/CrossModalData/SaveData/Project/NIPS19/AttackHash_SSAH_Flickr/Savecode/'

Flickr_Mean = h5py.File(Param['Meanpath'], 'r')
Param['meanpix'] = Flickr_Mean['Mean'][:].astype(np.float32)
Flickr_Mean.close()

Flickr = LoadData(Param['DATA_DIR'])

Img = h5py.File(Param['Imgpath'], 'r')
Param['Img_query'] = Img['ImgQuery'][:]
Param['Img_train'] = Img['ImgTrain'][:]
Param['Img_retrieval'] = Img['ImgDataBase'][:]

Tag = h5py.File(Param['Tagpath'], 'r')
Param['Txt_query'] = Tag['TagQuery'][:].astype(np.float32)
Param['Txt_train'] = Tag['TagTrain'][:].astype(np.float32)
Param['Txt_retrieval'] = Tag['TagDataBase'][:].astype(np.float32)

Lab = h5py.File(Param['Labpath'], 'r')
Param['Lab_query'] = Lab['LabQuery'][:].astype(np.float32)
Param['Lab_train'] = Lab['LabTrain'][:].astype(np.float32)
Param['Lab_retrieval'] = Lab['LabDataBase'][:].astype(np.float32)

Param['dimText'] = Param['Txt_train'].shape[1]
Param['num_train'] = Param['Txt_train'].shape[0]
Param['numClass'] = Param['Lab_train'].shape[1]
Param['image_size'] = 224
#========================================================================================#
Param['Results'] = '/home/chao/HashWorks/NIPS19/AttackHash_SSAH_Flickr/LogText'

Param['bit'] = 32

Param['beta'] = 1
Param['gamma'] = 1
Param['eta'] = 1
Param['delta'] = 1

Param['save_freq'] = 5
Param['batch_size'] = 128

Param['SEMANTIC_EMBED'] = 512


Param['lr_img'] = 0.0001 #0.0001
Param['lr_txt'] = 0.01 #0.001
Param['lr_lab'] = 0.01          #0.0001
Param['lr_dis'] = 0.01

Param['lr_adv_I'] = 0.5#            0.1
Param['lr_adv_T'] = 0.002#           0.001
Param['decay'] = 0.1         #0.5  #0.9
Param['decay_steps'] = 10
