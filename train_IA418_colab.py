# -*- coding: utf-8 -*-
# prueba de training from scratch con el dataset IA418_TF26 (50 imágenes con data augmentation)
import os
import logging #añadido 21dic2019 para sacar los prints() con los "losses" porque estaba escondido entre los warnings de los gradientes
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#device_count={'GPU': 0}

logging.basicConfig(
    filename='train.log', filemode='w', 
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.error('Log de la ejecución train_IA418_deep.py')

import tensorflow as tf
#tf.enable_eager_execution() eliminado en TF2
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import cv2
import os
import argparse
import json
import glob

from yolo.net import Yolonet
from yolo.train import train_fn
from yolo.frontend import YoloDetector
from yolo.utils.box import draw_boxes
from yolo.dataset.generator import BatchGenerator
from yolo.utils.utils import download_if_not_exists

argparser = argparse.ArgumentParser(
    description='train yolo-v3 network with training from IA418_TF26 50 images dataset 16dic2019')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/IA418_train1_colab.json",  # configuración de archivos y recursos (batch size...) para deep.teisa.unican.es
    help='config file')



if __name__ == '__main__':
    args = argparser.parse_args()
    with open(args.config) as data_file:    
        config = json.load(data_file)  #los valores de config se sacan de aquí

    # Download if not exits weight file
    download_if_not_exists(config["pretrained"]["darknet_format"],
                           "https://pjreddie.com/media/files/yolov3.weights")
    
    # 1. create generator
    train_ann_fnames = glob.glob(os.path.join(config["train"]["train_annot_folder"], "*.xml"))
    valid_ann_fnames = glob.glob(os.path.join(config["train"]["valid_annot_folder"], "*.xml"))
    #print("train_ann_fnames=" + train_ann_fnames)
    #print("valid_ann_fnames=" + valid_ann_fnames)
    print("cuantos ficheros XML: " , len(train_ann_fnames), len(valid_ann_fnames))
    train_generator = BatchGenerator(train_ann_fnames,
                                     config["train"]["train_image_folder"],
                                     batch_size=config["train"]["batch_size"],
                                     labels=config["model"]["labels"],
                                     anchors=config["model"]["anchors"],
                                     min_net_size=config["train"]["min_size"],
                                     max_net_size=config["train"]["max_size"],
                                     jitter=config["train"]["jitter"],
                                     shuffle=True)

    valid_generator = BatchGenerator(valid_ann_fnames,
                                       config["train"]["valid_image_folder"],
                                       batch_size=config["train"]["batch_size"],
                                       labels=config["model"]["labels"],
                                       anchors=config["model"]["anchors"],
                                       min_net_size=config["model"]["net_size"],
                                       max_net_size=config["model"]["net_size"],
                                       jitter=False,
                                       shuffle=False)
    
    print(train_generator.steps_per_epoch)
    
    # 2. create model
    model = Yolonet(n_classes=len(config["model"]["labels"]))
    model.load_weights(config["pretrained"]["keras_format"]) #intento de partir el entrenamiento de los pesos anteriores, se notará porque empezará con losses=6 aprox
    #model.load_darknet_params(config["pretrained"]["darknet_format"], skip_detect_layer=True)
    
 
    # 4. training
    train_fn(model,
             train_generator,
             valid_generator,
             learning_rate=config["train"]["learning_rate"],
             save_dname=config["train"]["save_folder"],
             num_epoches=config["train"]["num_epoch"])

    # 5. prepare sample images
    img_fnames = glob.glob(os.path.join(config["train"]["train_image_folder"], "*.*"))
    imgs = [cv2.imread(fname)[:,:,::-1] for fname in img_fnames]

    # 6. create new model & load trained weights
    model = Yolonet(n_classes=len(config["model"]["labels"]))
    model.load_weights(os.path.join(config["train"]["save_folder"], "weights.h5"))
    detector = YoloDetector(model)
 
    # 7. predict & plot
    boxes, labels, probs = detector.detect(imgs[0], config["model"]["anchors"])
    image = draw_boxes(imgs[0], boxes, labels, probs, class_labels=config["model"]["labels"])
    plt.imshow(image)
    plt.show()


