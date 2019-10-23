# NeurIPS2019 Cross-Modal Learning with Adversarial Samples

Step1:
======
  * Creat /Flickr-25K/ to save Dataset;  
    Download Img.h5, Img.h5, Lab.h5, and Mean.hs from [here](https://drive.google.com/drive/folders/1DcgBfKRoM8dCglaOamweQu6D6CynW7B_) and save to /Flickr-25K;  
  * Creat /model/ to save CNNF network;  
    Download imagenet-vgg-f.mat from [here](https://drive.google.com/drive/folders/1bbVTWN8IVMxchM2-xnwZRhoWb-Cj6_gj);
  * Creat /AttackHash_SSAH_Flickr/AdversarialSamples/ to save the learned Adversarial Samples;  
  * Creat /AttackHash_SSAH_Flickr/NoiseSamples/ to save the learned Perturbation;  
  * Creat /AttackHash_SSAH_Flickr/checkpoint to save checkpoint for tensorflow models;  
  * Creat /AttackHash_SSAH_Flickr/Savecode/ to save the learned hash codes;  
  * Creat AttackHash_SSAH_Flickr/LogText for a log print;  
  
Step2:
======
  * run demo_train_BasicModel.py to train a basic deep cross-modal hash network (CVPR18_SSAH);  
  * run demo_learn_AdversarialSampls.py to learn adversarial samples;
  * run demo_AdversarialTraining.py to finetuning the basic model;  
