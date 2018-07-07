
import tensorflow as tf
from lib.networks.factory import get_network
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


if __name__ == '__main__':
	# init session
    config = tf.ConfigProto(allow_soft_placement=True)
    #sess = tf.Session(config=config)
    net = get_network("VGGnet_test")
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state("/home/wagston/text-detection-ctpn/checkpoints/")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
        
        #for n in sess.graph.as_graph_def().node:
        #    print(n.name)

        graph_def = sess.graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess,
                                                                 graph_def,
                                                                 ['Reshape_2', 'rpn_bbox_pred/Reshape_1'])
        graph_io.write_graph(output_graph_def,
                         'exported_model',
                         'vgg_exported.pb',
                         as_text=False)