# Cross-Modal Learning with Adversarial Samples (NeurIPS2019)

Installation
=====
Tensorflow==1.12.0;   
Python=Python 3.6.8;  

Step1:
======
  * Creat __/Flickr-25K/__ to save Dataset;  
    Download __Img.h5, Img.h5, Lab.h5, and Mean.h5__ from [here](https://drive.google.com/drive/folders/1DcgBfKRoM8dCglaOamweQu6D6CynW7B_) and save to /Flickr-25K;  
  * Creat /model/ to save CNNF network;  
    Download __imagenet-vgg-f.mat__ from [here](https://drive.google.com/drive/folders/1bbVTWN8IVMxchM2-xnwZRhoWb-Cj6_gj);
  * Creat __/AttackHash_SSAH_Flickr/AdversarialSamples/__ to save the learned Adversarial Samples;  
  * Creat __/AttackHash_SSAH_Flickr/NoiseSamples/__ to save the learned Perturbation;  
  * Creat __/AttackHash_SSAH_Flickr/checkpoint/__ to save checkpoint for tensorflow models;  
  * Creat __/AttackHash_SSAH_Flickr/Savecode/__ to save the learned hash codes;  
  * Creat __AttackHash_SSAH_Flickr/LogText__/ for a log print;  
  
Step2:
======
  * run __demo_train_BasicModel.py__ to train a basic deep cross-modal hash network ([SSAH_CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Self-Supervised_Adversarial_Hashing_CVPR_2018_paper.pdf));  
  * run __demo_learn_AdversarialSampls.py__ to learn adversarial samples;
  * run __demo_AdversarialTraining.py__ to finetuning the basic model;  
