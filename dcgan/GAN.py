import os
import time
import tensorflow as tf
import numpy as np

from ops import *
from utils import *

class GAN(object):
    model_name = "GAN"     # name for checkpoint

    def __init__(self, sess, args):
        self.sess = sess
        self.dataset = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.z_dim = args.z_dim  # dimension of noise-vector

        self.sample_num = args.sample_num  # number of generated images to be saved
        self.test_num = args.test_num

        self.img_size = args.img_size

        # train
        self.learning_rate = 0.0002
        self.beta1 = 0.5


        if self.dataset == 'mnist' :
            self.c_dim = 1
            self.data_X = load_mnist(size=self.img_size)

        elif self.dataset == 'ciar10' :
            self.c_dim = 3
            self.data_X = load_cifar10(size=self.img_size)

        else :
            self.c_dim = 3
            self.data_X = load_data(dataset_name=self.dataset, size=self.img_size)


        # get number of batches for a single epoch
        self.num_batches = len(self.data_X) // self.batch_size

    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            ch = 32

            # x = conv(x, channels=ch, kernel=5, stride=2, pad=2, scope='conv_0')
            # x = lrelu(x)

            for i in range(5):

                # ch : 64 -> 128 -> 256 -> 512 -> 1024
                # size : 32 -> 16 -> 8 -> 4 -> 2

                x = conv(x, channels=ch*2, kernel=5, stride=2, pad=2, sn=True, scope='conv_'+str(i+1))
                x = batch_norm(x, is_training, scope='batch_'+str(i))
                x = lrelu(x)

                ch = ch * 2

            # [bs, 4, 4, 1024]

            x = flatten(x)
            x = fully_conneted(x, 1, sn=True)

            return x

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            ch = 1024

            x = fully_conneted(z, ch)
            x = relu(x)
            x = tf.reshape(x, [-1, 1, 1, ch])

            for i in range(5):

                # ch : 512 -> 256 -> 128 -> 64 -> 32
                # size : 2 -> 4 -> 8 -> 16 -> 32

                x = deconv(x, channels=ch//2, kernel=5, stride=2, scope='deconv_'+str(i+1))
                x = batch_norm(x, is_training, scope='batch_'+str(i))
                x = relu(x)
                ch = ch // 2

            x = deconv(x, channels=self.c_dim, kernel=5, stride=2, scope='generated_image')
            # [bs, 64, 64, c_dim]

            x = tanh(x)

            return x

    def build_model(self):

        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.c_dim], name='real_images')

        # noises
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')

        """ Loss Function """
        # output of D for real images
        D_real_logits = self.discriminator(self.inputs)

        # output of D for fake images
        G = self.generator(self.z)
        D_fake_logits = self.discriminator(G, reuse=True)

        # get loss for discriminator
        self.d_loss = discriminator_loss(real=D_real_logits, fake=D_fake_logits)

        # get loss for generator
        self.g_loss = generator_loss(fake=D_fake_logits)

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.fake_images = self.generator(self.z, is_training=False, reuse=True)

        """ Summary """
        self.d_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)



    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx*self.batch_size : (idx+1)*self.batch_size]

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])

                train_feed_dict = {
                    self.inputs : batch_images,
                    self.z : batch_z
                }

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(counter, self.print_freq) == 0:
                    samples = self.sess.run(self.fake_images, feed_dict={self.z: self.sample_z})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))

                    sample_dir = os.path.join(self.sample_dir, self.model_dir)
                    check_folder(sample_dir)

                    save_images(samples[:manifold_h * manifold_w, :, :, :],
                                [manifold_h, manifold_w],
                                './' + sample_dir + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(epoch, idx))

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            # self.save(self.checkpoint_dir, counter)

            # show temporal results
            # self.visualize_results(epoch)

        # save model for final step
        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

        sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(sample_dir)

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    sample_dir + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset, self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name+'.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        """ random condition, random noise """

        for i in range(self.test_num) :
            z_sample = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample})

            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :],
                        [image_frame_dim, image_frame_dim],
                        result_dir + '/' + self.model_name + '_test_all_classes_{}.png'.format(i))
