# -*- coding: utf-8 -*-

from yolo.darknet53 import Darknet53
from keras.models import load_model


def get_darknet_eager():
    from yolo import DARKNET_WEIGHTS
    from yolo.weights import WeightReader
    darknet_eager = Darknet53()
    reader = WeightReader(DARKNET_WEIGHTS)
    reader.load_weights(darknet_eager)
    return darknet_eager

def get_darknet_keras():
    from yolo import PROJECT_ROOT
    import os
    model = load_model(os.path.join(PROJECT_ROOT, "darknet53_weights.h5"))
    return model

if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)
    images = np.random.randn(1, 224, 224, 3)

#     darknet = get_darknet_keras()
#     ys = darknet.predict(images)
#     print(ys.shape)
#     np.save("ys_keras", ys)

    import tensorflow as tf
    tf.enable_eager_execution()
    darknet_eager = get_darknet_eager()
    ys = darknet_eager(tf.constant(images.astype(np.float32)))
    ys_eager = ys.numpy()
    ys_keras = np.load("ys_keras.npy")
    print(np.allclose(ys_eager, ys_keras))

# #     [ 0.03592291 -0.03602274 -0.03412793  0.03506648]
# #     [ 0.03353848  0.00134301 -0.02568628 -0.26286137]



    




