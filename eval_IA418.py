# -*- coding: utf-8 -*-
# prueba de inferencia con el entrenamiento del dataset IA418
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device_count={'GPU': 0}

import tensorflow as tf
import argparse


argparser = argparse.ArgumentParser(
    description='inferencia con el entrenamiento de IA418')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/IA418_eval.json",
    help='config file')

argparser.add_argument(
    '-s',
    '--save_dname',
    default=None)

argparser.add_argument(
    '-t',
    '--threshold',
    type=float,
    default=0.5)


if __name__ == '__main__':
    from yolo.config import ConfigParser
    args = argparser.parse_args()
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model()
    evaluator, _ = config_parser.create_evaluator(model)

    score = evaluator.run(threshold=args.threshold,
                          save_dname=args.save_dname)
    
    print(score)

