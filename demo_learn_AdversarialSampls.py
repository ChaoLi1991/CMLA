import os
import time
from PIL import Image
from ops import *
from tqdm import tqdm
import scipy.io as sio
from ops import calc_neighbor
from Module import SSAH
os.environ['CUDA_VISIBLE_DEVICES'] = select_gpu

gpuconfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction))

if not os.path.exists(Param['Results']):
    os.makedirs(Param['Results'])

TimeStamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ".txt"
pathLog = os.path.join(Param['Results'], TimeStamp)
Log_Txt = open(pathLog, 'a')


class Dummy:
    pass

env = Dummy()

def _AdervarialSampleLearn(Max_Iter, Image, Text, Label):

    index = np.arange(0,Param['num_adv'], 1)

    Adv_imgs = np.zeros([Param['num_adv'], 224, 224, 3], dtype=np.float32)
    Noi_imgs = np.zeros([Param['num_adv'], 224, 224, 3], dtype=np.float32)
    Adv_txts = np.zeros([Param['num_adv'], Param['dimText']], dtype=np.float32)
    Noi_txts = np.zeros([Param['num_adv'], Param['dimText']], dtype=np.float32)
    Adv_labs = np.zeros([Param['num_adv'], Param['numClass']], dtype=np.float32)


    Num_adv = len(Image)
    for batch in tqdm(range(Num_adv // env.batch_size + 1), ascii=True, desc="Train Attack"):

        ind = index[batch*env.batch_size : min((batch + 1) * env.batch_size, Num_adv)]
        ind_in = range(batch*env.batch_size, min((batch + 1) * env.batch_size, Num_adv))

        if len(ind) != env.batch_size:
            ind = index[-env.batch_size:]
            ind_in = range(len(index)-env.batch_size, len(index))

        image = Flickr.loadimg(Image[ind]).astype(np.float32)
        text = Text[ind,:]
        label = Label[ind, :]

        Adv_labs[ind_in, :] = label


        image = image
        text_exp = text[:, np.newaxis, :, np.newaxis]
        S = calc_neighbor(label, label)

        Cl_Hash_I = env.SSAH._generate_code(image, "image", True)
        Cl_Hash_T = env.SSAH._generate_code(text, "text")

        # =============Text============== #
        feed_dict_ti = {
            env.x_fixed: image,
            env.y_fixed: text_exp,
            env.CleHsh_I: Cl_Hash_I,
            env.CleHsh_T: Cl_Hash_T,
            env.eps_T: 1,
            env.Sim: S}

        # reset the noise before every iteration
        env.sess.run([env.ynoise.initializer])

        for iter in range(Max_Iter):

            _, adv_txt, ynoise = env.sess.run([env.Op_adv_train_T, env.adv_text, env.ynoise],
                                                           feed_dict=feed_dict_ti)
            sess.run(clip_op_t)

            Adv_txts[ind_in, :] = np.clip(np.squeeze(ynoise) + np.squeeze(text_exp), 0.0, 1.0)
            Noi_txts[ind_in, :] = np.squeeze(ynoise)

            if iter % 20 == 0:
                Loss_T, l2norm_T = env.sess.run([env.loss_ti0, env.l2norm_T], feed_dict=feed_dict_ti)
                print('[*] Iter: %d---Loss_T: %f---Indicator_P: %f' % (iter, Loss_T, l2norm_T))


    # =============Image============== #
        feed_dict_it = {
                env.x_fixed: image,
                env.y_fixed: text_exp,
                env.CleHsh_I: Cl_Hash_I,
                env.CleHsh_T: Cl_Hash_T,
                env.eps_I: 1,
                env.Sim: S }
            # reset the noise before every iteration
        env.sess.run([env.xnoise.initializer])

        for iter in range(Max_Iter):

            _, adv_img, xnoise = env.sess.run([env.Op_adv_train_I, env.adv_image, env.xnoise],
                                                          feed_dict=feed_dict_it)
            sess.run(clip_op_i)

            Adv_imgs[ind_in, :, :, :] = np.clip(xnoise + image, 0.0, 255.0)
            Noi_imgs[ind_in, :, :, :] = xnoise

            if iter % 20 == 0:
                Loss_I, l2norm_I = env.sess.run([env.loss_it0, env.l2norm_I], feed_dict=feed_dict_it)
                print('[*] Iter: %d---Loss_I: %f---Indicator_P: %f' % (iter, Loss_I, l2norm_I))


    return Adv_imgs, Noi_imgs, Adv_txts, Noi_txts, Adv_labs, index


def _Save_Adverarial_sample(adv_i, x_noise, adv_t, y_noise, adv_l, adv_itm, Iters):

    print('\n[*] Saving Adversarial Samples')
    print('\n[*] Saving Adversarial Samples', sep="\n", file=Log_Txt)

    # Saving adversarial samples
    Adv_save_path = Param['adv_dir'] + str(Param['bit']) + '_' + adv_itm + '_' + str(Iters) + '.h5'
    Adv = h5py.File(Adv_save_path, 'w')
    Adv.create_dataset('AdvImg', data=adv_i)
    Adv.create_dataset('AdvTag', data=adv_t)
    Adv.create_dataset('AdvLab', data=adv_l)
    Adv.close()

    print('\n[*] Saving Adversarial Images')
    print('\n[*] Saving Adversarial Images', sep="\n", file=Log_Txt)

    # Saving adversarial images
    for idx in range(len(adv_i)):
        adversary_image = np.array(adv_i[idx, :, :, :]).astype("uint8").reshape([224, 224, 3])
        im = Image.fromarray(adversary_image)
        path1 = "%s_%s_%s" % (str(Param['bit']), adv_itm, str(Iters))
        if not os.path.exists(os.path.join(Param['adv_dir'], path1)):
            os.makedirs(os.path.join(Param['adv_dir'], path1))
        path = os.path.join(Param['adv_dir'], path1, str(idx)+".jpg")
        im.save(path)

    print('\n[*] Saving Noise Images')
    print('\n[*] Saving Noise Images', sep="\n", file=Log_Txt)

    # Saving image noises
    for idx in range(len(x_noise)):
        noise_image = np.array(x_noise[idx, :, :, :]).astype("uint8").reshape([224, 224, 3])
        im = Image.fromarray(noise_image)
        path_img_1 = "%s_%s_%s" % (str(Param['bit']), adv_itm, str(Iters))

        if not os.path.exists(os.path.join(Param['noi_dir'], path_img_1)):
            os.makedirs(os.path.join(Param['noi_dir'], path_img_1))
        path_img = os.path.join(Param['noi_dir'], path_img_1, str(idx)+".jpg")
        im.save(path_img)

    print('\n[*] Saving Noise Texts')
    print('\n[*] Saving Noise Texts', sep="\n", file=Log_Txt)

    # Saving text noises
    path_txt_1 = "%s_%s_%s.h5" % (str(Param['bit']), adv_itm, str(Iters))
    path_txt = os.path.join(Param['noi_dir'], path_txt_1)

    noise_text = h5py.File(path_txt, 'w')
    noise_text.create_dataset('noi_txt', data=y_noise)
    noise_text.close()


with tf.Session(config=gpuconfig) as sess:
    env.sess = sess
    with tf.variable_scope("attack", reuse=tf.AUTO_REUSE) as scope:
        env.batch_size = Param['batch_size']
        env.BIT = Param['bit']
        env.x_fixed = tf.placeholder(tf.float32, [None, Param['image_size'], Param['image_size'], 3], name='image_input_fixed')
        env.y_fixed = tf.placeholder(tf.float32, [None, 1, Param['dimText'], 1], name='text_input_fixed')
        env.CleHsh_I = tf.placeholder(tf.float32, [None, Param['bit']], name='CleanHashcodes_Image')
        env.CleHsh_T = tf.placeholder(tf.float32, [None, Param['bit']], name='CleanHashcodes_Text')
        env.eps_I = tf.placeholder(tf.float32, (), name='eps_I')
        env.eps_T = tf.placeholder(tf.float32, (), name='eps_T')
        env.Sim = tf.placeholder(tf.float32, [None, Param['batch_size']], name='Sim_adv')
        env.SSAH = SSAH(sess)

        xshape = [Param['batch_size'], 224, 224, 3]
        yshape = [Param['batch_size'], 1, Param['dimText'], 1]
        env.xnoise = tf.get_variable('xnoise', xshape, tf.float32, initializer=tf.zeros_initializer, trainable=True)
        env.ynoise = tf.get_variable('ynoise', yshape, tf.float32, initializer=tf.zeros_initializer, trainable=True)

        env.adv_image = tf.clip_by_value(env.x_fixed + env.xnoise, 0.0, 255.0)
        env.adv_image_input = env.adv_image - tf.constant(np.repeat(Param['meanpix'][np.newaxis, :, :, :], Param['batch_size'], axis=0))

        # ============================Text============================#
        env.adv_text = tf.clip_by_value(env.y_fixed + env.ynoise, 0.0, 1.0)


        with tf.variable_scope("SSAH_model") as scope2:
            env.AdvHash_I, env.AdvHash_T = env.SSAH._build_model(env.adv_image_input, env.adv_text)


        # ========================Image============================#
        Item_it = tf.matmul(tf.squeeze(env.AdvHash_I), tf.squeeze(env.CleHsh_T), transpose_b=True)
        Item_ii = tf.matmul(tf.squeeze(env.AdvHash_I), tf.squeeze(env.CleHsh_I), transpose_b=True)
        env.loss_it0 = tf.reduce_mean(tf.multiply(env.Sim, Item_it) + tf.log(1.0 + tf.exp(-Item_it)))
        env.loss_it1 = tf.reduce_mean(tf.abs(env.x_fixed - env.adv_image))
        env.loss_it2 = tf.reduce_mean(tf.log(1.0 + tf.exp(Item_ii)) - tf.multiply(env.Sim, Item_ii))
        env.Loss_I = env.loss_it0 + env.eps_I * env.loss_it1 + env.loss_it2

            # ========================= Indicator ==================================#(0)
        norm_I = tf.reduce_sum(tf.square(env.adv_image/255.0 - env.x_fixed/255.0), axis=[1,2,3])
        env.l2norm_I = tf.reduce_mean(tf.sqrt(tf.divide(norm_I, Param['image_size']*Param['image_size']*3)))

        # ======================= Text ===========================#
        Item_ti = tf.matmul(tf.squeeze(env.AdvHash_T), tf.squeeze(env.CleHsh_I), transpose_b=True)
        Item_tt = tf.matmul(tf.squeeze(env.AdvHash_T), tf.squeeze(env.CleHsh_T), transpose_b=True)
        env.loss_ti0 = tf.reduce_mean(tf.multiply(env.Sim, Item_ti) + tf.log(1.0 + tf.exp(-Item_ti)))
        env.loss_ti1 = tf.reduce_mean(tf.abs(env.y_fixed - env.adv_text))
        env.loss_ti2 = tf.reduce_mean(tf.log(1.0 + tf.exp(Item_tt)) - tf.multiply(env.Sim, Item_tt))
        env.Loss_T = env.loss_ti0 + env.eps_T * env.loss_ti1 + env.loss_ti2


            # ============================= Indicator ============================#(0)
        norm_T = tf.reduce_sum(tf.square(tf.squeeze(env.adv_text) - tf.squeeze(env.y_fixed)))
        env.l2norm_T = tf.reduce_mean(tf.sqrt(tf.divide(norm_T, Param['dimText'])))

        # Variables
        variable_names = [v.name for v in tf.trainable_variables()]
        for variable in variable_names:
            print(variable)
            print(variable, sep="\n", file=Log_Txt)

        opt_adv_i = tf.train.AdamOptimizer(learning_rate=Param['lr_adv_I'])
        opt_adv_t = tf.train.AdamOptimizer(learning_rate=Param['lr_adv_T'])
        env.Op_adv_train_I = opt_adv_i.minimize(env.Loss_I, var_list= [env.xnoise])
        env.Op_adv_train_T = opt_adv_t.minimize(env.Loss_T, var_list= [env.ynoise])
        clip_op_i = tf.assign(env.xnoise, tf.clip_by_value(env.xnoise, 0.0, 10))
        clip_op_t = tf.assign(env.ynoise, tf.clip_by_value(env.ynoise, 0.0, 0.01))


        Epochs_BasicTrain = 10 #10
        Epochs_FinetTrain = 5
        continueTrainAdvModel = False
        AdversarialLearning = True
        Evluation_on_adver = True

        Iters = 500  # {100, 200, 500, 1000}

        print('\nInitializing graph')
        print('\nInitializing graph', sep="\n", file=Log_Txt)
        init = tf.global_variables_initializer()
        env.saver = tf.train.Saver()
        env.sess.run(init)



            # ============ Adversarial Learning ====================== #
        if AdversarialLearning:
            print('\n[*] Learn Adversarial Samples...')
            print('\n[*] Learn Adversarial Samples...', sep="\n", file=Log_Txt)

            if continueTrainAdvModel:
                print('\n[*] Loading Pretained Adversarial Model...')
                print('\n[*] Loading Pretained Adversarial Model...', sep="\n", file=Log_Txt)

                Advmodel_dir = "%s_%s_%s" % (Param['dataset_name'], Param['bit'], 'AdvTrain')
                AdvModel_checkpoint_dir = os.path.join(Param['checkpoint_dir'], Advmodel_dir)
                env.SSAH._load(env.saver, AdvModel_checkpoint_dir)

            else:
                print('\n[*] Loading Pretrained Basic Model...')
                print('\n[*] Loading Pretrained Basic Model...', sep="\n", file=Log_Txt)
                model_dir = "%s_%s_%s" % (Param['dataset_name'], Param['bit'], 'BasTrain')
                checkpoint_dir = os.path.join(Param['checkpoint_dir'], model_dir)
                env.SSAH._load(env.saver, checkpoint_dir)


            train_query = ['QuerySet']
            adv_itm = 'QuerySet'
            print('\n[*] Training Distangle model on {0} with {1} iterations.' .format(adv_itm, Iters))
            print('\n[*] Training Distangle model on {0} with {1} iterations.' .format(adv_itm, Iters),
                                                                                sep="\n", file=Log_Txt)

            Param['num_adv'] = 2000
            IMAGE = Param['Img_query']
            TEXT = Param['Txt_query']
            LABEL = Param['Lab_query']

            adv_i, x_noise, adv_t, y_noise, adv_l, index = _AdervarialSampleLearn(Iters, IMAGE, TEXT, LABEL)

            print('\n[*] Defense to Adversarial Query Samples...')
            print('\n[*] Defense to Adversarial Query Samples...', sep="\n", file=Log_Txt)

            MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T, \
            Hashcodes_Query_Image, Hashcodes_Query_Text, Hashcodes_Retrieval_Image, Hashcodes_Retrieval_Text \
                = env.SSAH._Evaluate(adv_i, "image",
                                     adv_t, "text",
                                     adv_l,
                                     Param['Lab_retrieval'])
            print('Image query  Text: %3.3f\n' % MAP_I2T)
            print('Image query  Text: %3.3f\n' % MAP_I2T, sep="\n", file=Log_Txt)
            print('Text  query Image: %3.3f\n' % MAP_T2I)
            print('Text  query Image: %3.3f\n' % MAP_T2I, sep="\n", file=Log_Txt)
            print('Image query Image: %3.3f\n' % MAP_I2I)
            print('Image query Image: %3.3f\n' % MAP_I2I, sep="\n", file=Log_Txt)
            print('Text  query  Text: %3.3f\n' % MAP_T2T)
            print('Text  query  Text: %3.3f\n' % MAP_T2T, sep="\n", file=Log_Txt)

            # Saving adversarial codes
            Adv_dataset_bit = "%s_%s_adv_%sIters_%s.mat" % (Param['dataset_name'], str(env.BIT), str(Iters), adv_itm)
            Adv_savePath = os.path.join(Param['code_dir'], Adv_dataset_bit)
            if os.path.exists(Adv_savePath):
                os.remove(Adv_savePath)
            sio.savemat(Adv_savePath, {'Qi': Hashcodes_Query_Image,     'Qt': Hashcodes_Query_Text,
                                       'Di': Hashcodes_Retrieval_Image, 'Dt': Hashcodes_Retrieval_Text,
                                       'retrieval_L': Param['Lab_retrieval'], 'query_L': Param['Lab_query']})


            print('\n[*] Save Adversarial Query Samples...')
            print('\n[*] Save Adversarial Query Samples...', sep="\n", file=Log_Txt)
            _Save_Adverarial_sample(adv_i, x_noise, adv_t, y_noise, adv_l, adv_itm, Iters)

            print('\n[@] Finish!')
            print('\n[@] Finish!', sep="\n", file=Log_Txt)

        if Evluation_on_adver:
            # ============== Loading pretrain model ============= #
            print('\n[*] Loading BasicModel model!')
            print('\n[*] Loading BasicModel model!', sep="\n", file=Log_Txt)
            model_dir = "%s_%s_%s" % (Param['dataset_name'], Param['bit'], 'BasTrain')
            checkpoint_dir = os.path.join(Param['checkpoint_dir'], model_dir)
            env.SSAH._load(env.saver, checkpoint_dir)

            print('\n[*] Defense to Adversarial Query Samples---before retrain')
            print('\n[*] Defense to Adversarial Query Samples---before retrain',
                                                                      sep="\n", file=Log_Txt)

                # ============== load adversarial Query samples ============== #
            test_iter_dir = str(Param['bit']) + '_QuerySet_' + str(Iters) + '.h5'
            test_adver_dir = os.path.join(Param['adv_dir'], test_iter_dir)
            Adver_Te = h5py.File(test_adver_dir, 'r')
            Adver_X_Te = Adver_Te['AdvImg'][:]
            Adver_Y_Te = Adver_Te['AdvTag'][:]
            Adver_L_Te = Adver_Te['AdvLab'][:]
            Adver_Te.close()

                # Adversarial testing before adversarial training
            MAP_I2T, MAP_T2I, MAP_I2I, MAP_T2T, _, _, _, _ \
                        = env.SSAH._Evaluate(Adver_X_Te, "image",
                                             Adver_Y_Te, "text",
                                             Adver_L_Te,
                                             Param['Lab_retrieval'])

            print('Image query  Text: %3.3f\n' % MAP_I2T)
            print('Image query  Text: %3.3f\n' % MAP_I2T, sep="\n", file=Log_Txt)
            print('Text  query Image: %3.3f\n' % MAP_T2I)
            print('Text  query Image: %3.3f\n' % MAP_T2I, sep="\n", file=Log_Txt)
            print('Image query Image: %3.3f\n' % MAP_I2I)
            print('Image query Image: %3.3f\n' % MAP_I2I, sep="\n", file=Log_Txt)
            print('Text  query  Text: %3.3f\n' % MAP_T2T)
            print('Text  query  Text: %3.3f\n' % MAP_T2T, sep="\n", file=Log_Txt)
