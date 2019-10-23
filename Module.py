from tqdm import tqdm
import os
from model import *
from utils.calc_hammingranking import calc_map

class SSAH(object):
    def __init__(self, sess):
        self.batch_size = Param['batch_size']
        self.var = {}

        self.query_x_list = Param['Img_query']
        self.train_x_list = Param['Img_train']
        self.retrieval_x_list = Param['Img_retrieval']
        self.query_L = Param['Lab_query']
        self.train_L = Param['Lab_train']
        self.retrieval_L = Param['Lab_retrieval']

        self.query_Y = Param['Txt_query']
        self.train_Y = Param['Txt_train']
        self.retrieval_Y = Param['Txt_retrieval']

        self.alpha = Param['alpha']
        self.beta = Param['beta']
        self.gamma = Param['gamma']
        self.eta = Param['eta']
        self.delta = Param['delta']

        self.Lr_img = Param['lr_img']
        self.Lr_txt = Param['lr_txt']
        self.Lr_lab = Param['lr_lab']
        self.Lr_dis = Param['lr_dis']
        self.meanpix = Param['meanpix']

        self.lab_net = lab_net
        self.img_net = img_net
        self.txt_net = txt_net
        self.dis_net_IL = dis_net_IL
        self.dis_net_TL = dis_net_TL

        self.mse_loss = mse_criterion

        self.numClass = Param['numClass']
        self.dimText = Param['dimText']
        self.checkpoint_dir = Param['checkpoint_dir']
        self.dataset_dir = Param['dataset_name']
        self.BIT = Param['bit']
        self.num_train = Param['num_train']
        self.batch_size = Param['batch_size']
        self.save_freq = Param['save_freq']

        self.decay = Param['decay']
        self.decay_steps = Param['decay_steps']
        self.sess = sess

    def _build_model(self, image_input, text_input):
        self.ph = {}
        self.ph['label_input'] = tf.placeholder(tf.float32, [None, 1, 1, self.numClass], name='label_input')
        self.ph['image_input'] = image_input
        self.ph['text_input'] = text_input
        self.ph['lr_txt'] = tf.placeholder(tf.float32, None, name='lr_txt')
        self.ph['lr_img'] = tf.placeholder(tf.float32, None, name='lr_img')
        self.ph['learn_rate'] = tf.placeholder(tf.float32, None, name='learn_rate')
        self.ph['Sim'] = tf.placeholder(tf.float32, [None, self.batch_size], name='Sim')
        self.ph['epoch'] = tf.placeholder(tf.float32, None, name='Pepoch')
        self.ph['L_batch'] = tf.placeholder(tf.float32, [None, self.numClass], name='L_batch')

        with tf.variable_scope("SSAH", reuse=tf.AUTO_REUSE):
            # Construct label network
            self.Hsh_L, self.Fea_L, self.Lab_L = self.lab_net(self.ph['label_input'], self.BIT, self.numClass, name="Lab_Network")

            # construct image network
            self.Hsh_I, self.Fea_I, self.Lab_I = self.img_net(self.ph['image_input'], self.BIT, self.numClass, name="Img_Network")

            # construct text network
            self.Hsh_T, self.Fea_T, self.Lab_T = self.txt_net(self.ph['text_input'], self.BIT, self.numClass, name="Txt_Network")

            # construct two discriminator networks
            self.isfrom_IL = self.dis_net_IL(self.Fea_I, name="disnet_IL")
            self.isfrom_L1 = self.dis_net_IL(self.Fea_L, name="disnet_IL")
            self.isfrom_TL = self.dis_net_TL(self.Fea_T, name="disnet_TL")
            self.isfrom_L2 = self.dis_net_TL(self.Fea_L, name="disnet_TL")
            # loss_D
            Loss_adver_IL = self.mse_loss(self.isfrom_IL, tf.zeros_like(self.isfrom_IL))
            Loss_adver_TL = self.mse_loss(self.isfrom_TL, tf.zeros_like(self.isfrom_TL))
            Loss_adver_L1 = self.mse_loss(self.isfrom_L1, tf.ones_like(self.isfrom_L1))
            Loss_adver_L2 = self.mse_loss(self.isfrom_L2, tf.ones_like(self.isfrom_L2))
            self.Loss_D = tf.div(Loss_adver_IL + Loss_adver_TL + Loss_adver_L1 + Loss_adver_L2, 4.0)

            # train lab_net
            theta_L = 1.0 / 2 * tf.matmul(self.Hsh_L, tf.transpose(self.Hsh_L))
            Loss_pair_Hsh_L = - tf.reduce_sum(tf.multiply(self.ph['Sim'], theta_L) - tf.log(1.0 + tf.exp(theta_L)))
            Loss_quant_L = self.mse_loss(self.Hsh_L, tf.sign(self.Hsh_L))
            Loss_label_L = self.mse_loss(self.Lab_L, self.ph['L_batch'])
            self.Loss_L = Param['gamma'] * Loss_pair_Hsh_L \
                          + Param['beta'] * Loss_quant_L \
                          + Param['eta'] * Loss_label_L

            # train img_net combined with lab_net
            theta_I = 1.0 / 2 * tf.matmul(self.Hsh_I, tf.transpose(self.Hsh_L))
            self.Loss_pair_Hsh_I = - tf.reduce_sum(
                tf.multiply(self.ph['Sim'], theta_I) - tf.log(1.0 + tf.exp(theta_I)))
            self.Loss_quant_I = self.mse_loss(self.Hsh_I, tf.sign(self.Hsh_I))
            self.Loss_label_I = self.mse_loss(self.Lab_I, self.ph['L_batch'])
            self.Loss_adver_I = self.mse_loss(self.isfrom_IL, tf.ones_like(self.isfrom_IL))
            self.Loss_I = Param['gamma'] * self.Loss_pair_Hsh_I \
                          + Param['beta'] * self.Loss_quant_I \
                          + Param['eta'] * self.Loss_label_I \
                          + Param['delta'] * self.Loss_adver_I

            # train txt_net combined with lab_net
            theta_T = 1.0 / 2 * tf.matmul(self.Hsh_T, tf.transpose(self.Hsh_L))
            self.Loss_pair_Hsh_T = - tf.reduce_sum(
                tf.multiply(self.ph['Sim'], theta_T) - tf.log(1.0 + tf.exp(theta_T)))
            self.Loss_quant_T = self.mse_loss(self.Hsh_T, tf.sign(self.Hsh_T))
            self.Loss_label_T = self.mse_loss(self.Lab_T, self.ph['L_batch'])
            self.Loss_adver_T = self.mse_loss(self.isfrom_TL, tf.ones_like(self.isfrom_TL))
            self.Loss_T = Param['gamma'] * self.Loss_pair_Hsh_T \
                          + Param['beta'] * self.Loss_quant_T \
                          + Param['eta'] * self.Loss_label_T \
                          + Param['delta'] * self.Loss_adver_T

            # train variable
            all_vars = tf.trainable_variables()
            self.lab_vars = [var for var in all_vars if 'Lab_Network' in var.name]
            self.img_vars = [var for var in all_vars if 'Img_Network' in var.name]
            self.txt_vars = [var for var in all_vars if 'Txt_Network' in var.name]
            self.dis_i_vars = [var for var in all_vars if 'disnet_IL' in var.name]
            self.dis_t_vars = [var for var in all_vars if 'disnet_IL' in var.name]

            # for i, t in enumerate(self.img_vars):
            #     print(i, t.name)
            # #
            # print('all')

            # Learning rate
            self.lr_lab = tf.train.exponential_decay(
                learning_rate=self.Lr_lab, global_step=self.ph['epoch'], decay_steps=Param['decay_steps'],
                                                                         decay_rate=Param['decay'], staircase=True)
            self.lr_img = tf.train.exponential_decay(
                learning_rate=self.Lr_img, global_step=self.ph['epoch'], decay_steps=Param['decay_steps'],
                                                                         decay_rate=Param['decay'], staircase=True)
            self.lr_txt = tf.train.exponential_decay(
                learning_rate=self.Lr_txt, global_step=self.ph['epoch'], decay_steps=Param['decay_steps'],
                                                                         decay_rate=Param['decay'], staircase=True)
            self.lr_dis = tf.train.exponential_decay(
                learning_rate=self.Lr_dis, global_step=self.ph['epoch'], decay_steps=Param['decay_steps'],
                                                                         decay_rate=Param['decay'], staircase=True)

            opt_l = tf.train.AdamOptimizer(self.ph['learn_rate'])
            gradient_l = opt_l.compute_gradients(self.Loss_L, var_list=self.lab_vars)
            self.train_l = opt_l.apply_gradients(gradient_l)

            opt_i = tf.train.AdamOptimizer(self.ph['learn_rate'])
            gradient_i = opt_i.compute_gradients(self.Loss_I, var_list=self.img_vars)
            self.train_i = opt_i.apply_gradients(gradient_i)

            opt_t = tf.train.AdamOptimizer(self.ph['learn_rate'])
            gradient_t = opt_t.compute_gradients(self.Loss_T, var_list=self.txt_vars)
            self.train_t = opt_t.apply_gradients(gradient_t)

            opt_d = tf.train.AdamOptimizer(self.ph['learn_rate'])
            gradient_d = opt_d.compute_gradients(self.Loss_D, var_list=self.dis_i_vars + self.dis_t_vars)
            self.train_dis = opt_d.apply_gradients(gradient_d)

            return self.Hsh_I, self.Hsh_T

    def _Train(self, Saver, Train_img, Train_txt, Train_lab, Epoch, Model=None, Iters=None):

        self.saver = Saver

        # Iterations
        for epoch in range(Epoch+1):
            self.epoch = epoch

            index = np.random.permutation(self.num_train)
            print ('\n[*] ++++++++Start Train++++++++')

            # Train Lab_Net
            print('\n[*] ++++++++Start Train lab_net++++++++')
            for idx in range(1):
                learn_rate = self.sess.run(self.lr_lab, feed_dict={self.ph['epoch']: epoch})
                self.train_lab_net(Train_lab, learn_rate, epoch, index)

            print('\n[*] ++++++++Start Train dis_net++++++++')
            for idx in range(3):
                learn_rate = self.sess.run(self.lr_dis, feed_dict={self.ph['epoch']: epoch})
                self.train_dis_net(Train_img, Train_txt, Train_lab, learn_rate)

            print('\n[*] ++++++++Starting Train txt_net++++++++')
            # Train Txt_Net
            for idx in range(1):
                learn_rate = self.sess.run(self.lr_txt, feed_dict={self.ph['epoch']: epoch})
                self.train_txt_net(Train_txt, Train_lab, learn_rate, epoch, index)

            print('\n[*] ++++++++Starting Train img_net++++++++')
            # Train Img_Net
            for idx in range(1):
                learn_rate = self.sess.run(self.lr_img, feed_dict={self.ph['epoch']: epoch})
                self.train_img_net(Train_img, Train_lab, learn_rate, epoch, index)

            if Model == 'BasicModel' and np.mod(epoch, self.save_freq)==0 and epoch>0:
                print('\n[*] Saving the training model')
                model_dir = "%s_%s_%s" % (Param['dataset_name'], Param['bit'], 'BasTrain')
                checkpoint_dir = os.path.join(Param['checkpoint_dir'], model_dir)
                self._save(checkpoint_dir, epoch, model_name=Model)

            if Model == 'FinetuneModel' and np.mod(epoch, self.save_freq)==0 and epoch>0:
                print('\n[*] Saving the Adversarial Model...')
                model_dir = "%s_%s_%s" % (Param['dataset_name'], Param['bit'], 'AdvTrain')
                checkpoint_dir = os.path.join(Param['checkpoint_dir'], model_dir, str(Iters))
                self._save(checkpoint_dir, epoch, model_name=Model)



    def train_lab_net(self, train_L, learn_rate, epoch, index):

        print ('update lab net' + ' using ' + str(epoch))
        num_train = len(train_L)
        for iter in tqdm(range(num_train // self.batch_size + 1), ascii=True, desc="Iter"):
            ind = index[iter * self.batch_size: min((iter + 1) * self.batch_size, num_train)]
            if len(ind) != self.batch_size:
                ind = index[-self.batch_size:]

            label = train_L[ind, :]
            label_input = label[:, np.newaxis, np.newaxis, :]
            S = calc_neighbor(label, label)  #

            _,  Hsh_L, Loss_L = self.sess.run([self.train_l, self.Hsh_L, self.Loss_L],
                                             feed_dict={self.ph['Sim']: S,
                                                        self.ph['L_batch']: label,
                                                        self.ph['learn_rate']: learn_rate,
                                                        self.ph['label_input']: label_input})

            if iter % 20 == 0:
                Loss_L = self.sess.run(self.Loss_L,
                                       feed_dict={self.ph['Sim']: S,
                                                  self.ph['L_batch']: label,
                                                  self.ph['label_input']: label_input})

                print('\n------------------------------------------------------------------------------------')
                print('[*] Epoch: {0}, Iter: {1}, Loss_L: {2:.5f}, Learn_Rate: {3:.10f}'
                      .format(self.epoch, iter, Loss_L, learn_rate))
                print('------------------------------------------------------------------------------------')


    def train_txt_net(self, train_Y, train_L, learn_rate, epoch, index):

        print ('update text net' + ' using ' + str(epoch))
        num_train = len(train_L)
        for iter in tqdm(range(num_train // self.batch_size + 1), ascii=True, desc="Iter"):
            ind = index[iter * self.batch_size: min((iter + 1) * self.batch_size, num_train)]
            if len(ind) != self.batch_size:
                ind = index[-self.batch_size:]

            text = train_Y[ind, :]
            text_input = text[:, np.newaxis, :, np.newaxis]
            label = train_L[ind, :]
            label_input = label[:, np.newaxis, np.newaxis, :]

            S = calc_neighbor(label, label)

            Hsh_T, Loss_T, _  = self.sess.run([self.Hsh_T, self.Loss_T, self.train_t],
                                               feed_dict={self.ph['Sim']: S,
                                               self.ph['L_batch']: label,
                                               self.ph['learn_rate']: learn_rate,
                                               self.ph['text_input']: text_input,
                                               self.ph['label_input']: label_input})

            if iter % 20 == 0:
                Loss_T = self.sess.run(self.Loss_T, feed_dict={self.ph['Sim']: S,
                                                               self.ph['L_batch']: label,
                                                               self.ph['text_input']: text_input,
                                                               self.ph['label_input']: label_input})
                print('\n------------------------------------------------------------------------------------')
                print('[*] Epoch: {0}, Iter: {1}, Loss_T: {2:.5f}, Learn_Rate: {3:.10f}'
                        .format(self.epoch, iter, Loss_T,learn_rate))
                print('------------------------------------------------------------------------------------')

    def train_img_net(self, train_x_list, train_L, learn_rate, epoch, index):
        print ('update image net' + ' using ' + str(epoch))
        num_train = len(train_L)
        for iter in tqdm(range(num_train // self.batch_size + 1), ascii=True, desc="Iter"): #
            ind = index[iter * self.batch_size: min((iter + 1) * self.batch_size, num_train)]
            if len(ind) != self.batch_size:
                ind = index[-self.batch_size:]


            image = Flickr.loadimg(train_x_list[ind]).astype(np.float32)
            image_input = image - np.repeat(self.meanpix[np.newaxis, :, :, :], self.batch_size, axis=0)
            label = train_L[ind, :]

            label_input = label[:, np.newaxis, np.newaxis, :]
            S = calc_neighbor(label, label)

            Hsh_I, Loss_I, _ = self.sess.run([self.Hsh_I, self.Loss_I, self.train_i],
                                                  feed_dict={self.ph['Sim']: S,
                                                             self.ph['L_batch']: label,
                                                             self.ph['learn_rate']: learn_rate,
                                                             self.ph['image_input']: image_input,
                                                             self.ph['label_input']: label_input})

            if iter % 20 == 0:
                Loss_I = self.sess.run(self.Loss_I,
                                       feed_dict={self.ph['Sim']: S,
                                                  self.ph['L_batch']: label,
                                                  self.ph['image_input']: image_input,
                                                  self.ph['label_input']: label_input})
                print('\n------------------------------------------------------------------------------------')
                print('[*] Epoch: {0}, Iter: {1}, Loss_I: {2:.5f}, Learn_Rate: {3:.10f}'
                      .format(self.epoch, iter, Loss_I, learn_rate))
                print('------------------------------------------------------------------------------------')


    def train_dis_net(self, train_x_list, train_Y, train_L, learn_rate):
        print('\n[*] Update dis_net')
        num_train = len(train_L)

        for iter in tqdm(range(num_train // self.batch_size), ascii=True, desc="Iter"):
            index = np.random.permutation(num_train)
            ind = index[0: self.batch_size]
            image = Flickr.loadimg(train_x_list[ind]).astype(np.float32)
            image_input = image - np.repeat(self.meanpix[np.newaxis, :, :, :], self.batch_size, axis=0)

            text = train_Y[ind, :]
            text_input = text[:, np.newaxis, :, np.newaxis]
            label = train_L[ind, :]
            label_input = label[:, np.newaxis, np.newaxis, :]

            self.train_dis.run(feed_dict={self.ph['image_input']: image_input,
                                          self.ph['text_input']: text_input,
                                          self.ph['label_input']: label_input,
                                          self.ph['learn_rate']: learn_rate})

            isfrom_IL = self.isfrom_IL.eval(feed_dict={self.ph['image_input']: image_input})
            isfrom_L1 = self.isfrom_L1.eval(feed_dict={self.ph['label_input']: label_input})
            isfrom_TL = self.isfrom_TL.eval(feed_dict={self.ph['text_input']: text_input})

            if iter % 20 == 0:
                Loss_Dis = self.sess.run(self.Loss_D, feed_dict={self.ph['image_input']: image_input,
                                                                 self.ph['text_input']: text_input,
                                                                 self.ph['label_input']: label_input,
                                                                 self.ph['learn_rate']: learn_rate})

                IsFrom_ = np.hstack((isfrom_IL, isfrom_L1, isfrom_TL))
                IsFrom  = np.hstack((np.zeros_like(isfrom_IL), np.ones_like(isfrom_L1), np.zeros_like(isfrom_TL)))
                erro, Acc = self.calc_isfrom_acc(IsFrom_, IsFrom)
                print('\n------------------------------------------------------------------------------------')
                print('[*] Epoch: {0}, Iter: {1}, Loss_D: {2:.5f}, Acc_D: {3:.5f}, Learn_Rate: {4:.10f}'
                           .format(self.epoch, iter, Loss_Dis, Acc, learn_rate))
                print('------------------------------------------------------------------------------------')



    def _generate_code(self, Modal, generate, Adversarial=False):

        num_data = Modal.shape[0]
        index = np.linspace(0, num_data - 1, num_data).astype(int)
        BIT = np.zeros([num_data, self.BIT], dtype=np.float32)
        if generate == "label":
            for iter in tqdm(range(num_data // self.batch_size + 1)):
                ind = index[iter * self.batch_size: min((iter + 1) * self.batch_size, num_data)]
                label = Modal[ind, :]
                Label = label[:, np.newaxis, np.newaxis, :]
                BIT[ind, :] = np.squeeze(self.Hsh_L.eval(feed_dict={self.ph['label_input']: Label}))

        elif generate == "image":
            if Adversarial == True or len(Modal.shape) == 4:
                num_data = len(Modal)
                index = np.linspace(0, num_data - 1, num_data).astype(int)
                BIT = np.zeros([num_data, self.BIT], dtype=np.float32)
                for iter in tqdm(range(num_data // self.batch_size + 1)):
                    ind = index[iter * self.batch_size: min((iter + 1) * self.batch_size, num_data)]
                    if len(ind) != self.batch_size:
                        ind = index[-self.batch_size:]
                    mean_pixel = np.repeat(self.meanpix[:, :, :, np.newaxis], len(ind), axis=3)
                    image = Modal[ind, :, :, :].astype(np.float32)
                    image_input = image - mean_pixel.astype(np.float32).transpose(3, 0, 1, 2)
                    BIT[ind, :] = np.squeeze(self.Hsh_I.eval(feed_dict={self.ph['image_input']: image_input}))

            else:
                for iter in tqdm(range(num_data // self.batch_size + 1)):
                    ind = index[iter * self.batch_size: min((iter + 1) * self.batch_size, num_data)]
                    if len(ind) != self.batch_size:
                        ind = index[-self.batch_size:]
                    mean_pixel = np.repeat(self.meanpix[:, :, :, np.newaxis], len(ind), axis=3)
                    image = Flickr.loadimg(Modal[ind]).astype(np.float32)
                    image_input = image - mean_pixel.astype(np.float32).transpose(3, 0, 1, 2)
                    BIT[ind, :] = np.squeeze(self.Hsh_I.eval(feed_dict={self.ph['image_input']: image_input}))

        elif generate == "text":
            for iter in tqdm(range(num_data // self.batch_size + 1)):
                ind = index[iter * self.batch_size: min((iter + 1) * self.batch_size, num_data)]
                if len(ind) != self.batch_size:
                    ind = index[-self.batch_size:]
                text = Modal[ind, :].astype(np.float32)
                text_input = text[:, np.newaxis, :, np.newaxis]
                BIT[ind, :] = np.squeeze(self.Hsh_T.eval(feed_dict={self.ph['text_input']: text_input}))

        else:
            print("Wrong Input!")
            os._exit(0)

        return np.sign(BIT)


    def _Evaluate(self, Modal_X, X_name, Modal_Y, Y_name, query_L, retrieval_L):

        Hashcodes_QX = self._generate_code(Modal_X, X_name)
        Hashcodes_QY = self._generate_code(Modal_Y, Y_name)
        Hashcodes_RX = self._generate_code(Param['Img_retrieval'], "image",)
        Hashcodes_RY = self._generate_code(Param['Txt_retrieval'], "text",)

        mapx2y = calc_map(Hashcodes_QX, Hashcodes_RY, query_L, retrieval_L)
        mapx2x = calc_map(Hashcodes_QX, Hashcodes_RX, query_L, retrieval_L)
        mapy2x = calc_map(Hashcodes_QY, Hashcodes_RX, query_L, retrieval_L)
        mapy2y = calc_map(Hashcodes_QY, Hashcodes_RY, query_L, retrieval_L)

        return mapx2y, mapy2x, mapx2x, mapy2y, Hashcodes_QX, Hashcodes_QY, Hashcodes_RX, Hashcodes_RY


    def calc_isfrom_acc(self, train_isfrom_, Train_ISFROM):
        erro = Train_ISFROM.shape[0] - np.sum(np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype(int))
        acc = np.divide(np.sum(np.equal(np.sign(train_isfrom_ - 0.5), np.sign(Train_ISFROM - 0.5)).astype('float32')), Train_ISFROM.shape[0])
        return erro, acc

    def _save(self, checkpoint_dir, step, model_name):

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name),
                        global_step = step)

    def _load(self, saver, checkpoint_dir):

        print (" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print (ckpt_name)
            print(" [*] Load SUCCESS")
            return True
        else:
            print(" [*] Load FAILED")
            return False

