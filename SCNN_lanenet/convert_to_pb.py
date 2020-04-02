import os
import os.path as ops
import argparse
import math
import tensorflow as tf
import glog as log
import cv2
try:
    from cv2 import cv2
except ImportError:
    pass

import lanenet_merge_model
import global_config
import lanenet_data_processor_test

saver = tf.train.Saver()

sess = tf.Session()

with sess.as_default():
    sess.run(tf.global_variables_initializer())
    saver.restore(sess=sess, save_path='model_culane-71-3/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000')

    tf.train.write_graph(sess.graph.as_graph_def(), "./model/", 'pb_model/scnn_lanenet/saved_model.pb', False)
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,output_node_names=['layer2/BiasAdd'])
    tf.train.write_graph(output_graph_def, "./model/", "graph.pbtxt",as_txt = False)