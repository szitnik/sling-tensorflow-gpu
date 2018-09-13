import tensorflow as tf

from tensorflow.python.client import device_lib

def get_available_gpus():

    # Tensorflow porabi vse resurse, ki so na voljo, ne toliko, kolikor rabi oziroma kar je alocirano prek upravljavca nalog.
    # To popravimo tako, da ze pri vzpostavitvi seje dolocimo, da se rama ne alocira vnaprej:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    

    local_device_protos = device_lib.list_local_devices(config)
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()