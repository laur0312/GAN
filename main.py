import os
import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim
from GAN import GAN
from DCGAN import DCGAN
from CDCGAN import CDCGAN
from WGAN import WGAN

flags = tf.app.flags
flags.DEFINE_integer('epochs', 25, 'Epochs to train.')
flags.DEFINE_string('model', "GAN", 'The name of model.')
flags.DEFINE_string('result_dir', 'results', 'Directory to save the generated images.')
flags.DEFINE_string('model_dir', 'models', 'Directory to save trained modes.')
FLAGS = flags.FLAGS


def main(_):
    pprint.PrettyPrinter().pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.result_dir):
        os.makedirs(FLAGS.result_dir)

    if not os.path.isdir(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        if FLAGS.model == 'GAN':
            gan = GAN(sess, result_dir=FLAGS.result_dir, model_dir=FLAGS.model_dir)
        elif FLAGS.model == 'DCGAN':
            gan = DCGAN(sess, result_dir=FLAGS.result_dir, model_dir=FLAGS.model_dir)
        elif FLAGS.model == 'CDCGAN':
            gan = CDCGAN(sess, result_dir=FLAGS.result_dir, model_dir=FLAGS.model_dir)
        elif FLAGS.model == 'WGAN':
            gan = WGAN(sess, result_dir=FLAGS.result_dir, model_dir=FLAGS.model_dir)
        else:
            print('Error! Invalid input.')
            exit(0)

        gan.build_model()
        show_all_variables()
        gan.train(FLAGS.epochs)


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


if __name__ == '__main__':
    tf.app.run()
