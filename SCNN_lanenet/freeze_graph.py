import tensorflow as tf

with tf.gfile.GFile('pb_model/scnn_lanenet/saved_model.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    
tf.train.write_graph(graph_def, '', 'pb_model/scnn_lanenet/frozen_inference_graph_face.pbtxt')