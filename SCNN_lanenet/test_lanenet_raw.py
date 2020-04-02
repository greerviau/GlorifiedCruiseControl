import os
import os.path as ops
import argparse
import math
import tensorflow as tf
import numpy as np
import glog as log
import cv2
import numpy as np
try:
    from cv2 import cv2
except ImportError:
    pass


import SCNN_lanenet.lanenet_merge_model  as lmm
import SCNN_lanenet.global_config as gc
import SCNN_lanenet.lanenet_data_processor_test as ldpt


CFG = gc.cfg
VGG_MEAN = [103.939, 116.779, 123.68]


class Lanenet():

    def __init__(self, weights_path, use_gpu,):

        test_dataset = ldpt.DataSet('SCNN_lanenet/image_list.txt', 1)
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH, 3])
        imgs = tf.map_fn(test_dataset.process_img_raw, self.input_tensor, dtype=tf.float32)
        
        phase_tensor = tf.constant('test', tf.string)

        net = lmm.LaneNet()
        self.binary_seg_ret, self.instance_seg_ret = net.test_inference(imgs, phase_tensor, 'lanenet_loss')
        initial_var = tf.global_variables()
        final_var = initial_var[:-1]
        self.saver = tf.train.Saver(final_var)
        # Set sess configuration
        if use_gpu:
            sess_config = tf.ConfigProto(device_count={'GPU': 1})
        else:
            sess_config = tf.ConfigProto(device_count={'GPU': 0})
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=sess_config)

        self.sess.run(tf.global_variables_initializer())
        print(weights_path)
        self.saver.restore(sess=self.sess, save_path=weights_path)    
    
    def run_lanenet(self, image):
        img_resized = cv2.resize(image, (CFG.TRAIN.IMG_WIDTH, CFG.TRAIN.IMG_HEIGHT))
        img_resized.astype(np.float32)
        img = np.subtract(img_resized, VGG_MEAN)

        instance_seg_image, existence_output = self.sess.run([self.binary_seg_ret, self.instance_seg_ret], feed_dict={self.input_tensor: [img]})

        output_frame = (instance_seg_image * 255)[0][:,:,1:4].astype(int)

        #cv2.imshow('out', output_frame*255)
        #cv2.waitKey(0)

        return output_frame


if __name__ == '__main__':
    
    lanenet = Lanenet('SCNN_lanenet/model_culane-71-3/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000', True)
    
    
    op_frame = lanenet.test_lanenet('SCNN_lanenet/test_images/frame_0.png')
    #print(op_frame.shape)
    print(op_frame.shape)
    #cv2.imwrite('lane_test.png', op_frame[:,:,1:4])
    cv2.imshow('OP', op_frame*255)
    cv2.waitKey(0)