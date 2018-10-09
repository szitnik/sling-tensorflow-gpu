# Python
import os
os.environ["CUDA_VISIBLE_DEVICES"]=“0"

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')

# Tensorflow porabi vse resurse, ki so na voljo, ne toliko, kolikor rabi oziroma kar je alocirano prek upravljavca nalog.
# To popravimo tako, da ze pri vzpostavitvi seje dolocimo, da se rama ne alocira vnaprej:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config= config)
print(sess.run(hello))