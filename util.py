import tensorflow as tf
import numpy as np
import os
import align.detect_face
from scipy import misc

def load_model(sess):

    dirname = os.path.dirname(__file__)

    saver = tf.train.import_meta_graph(dirname + '/trained_params/model-20170512-110547.meta', clear_devices=True)#load graph
    saver.restore(sess, dirname + '/trained_params' + '/model-20170512-110547.ckpt-250000') #load weights

    x = tf.get_default_graph().get_tensor_by_name('input:0') # input placeholder
    out = tf.get_default_graph().get_tensor_by_name("embeddings:0") # output tensor
    # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    return x,out

def distance(a,b):
    return np.sqrt(np.sum(np.square(a-b)))

def get_embedding(sess, input, output, image):

    embedding = sess.run(output,feed_dict={input:image})
    if(image.shape[0]==1):
        return embedding.reshape(128)

    return embedding

def create_dataset(sess, mtcnn, input, output, base_dir):

    files = os.listdir(base_dir)
    batch_size = 50

    names = []
    images = []
    embeddings = None
    cnt = 1
    flag = 1

    for file in files:
        if os.path.isfile(os.path.join(base_dir,file)):

            name = file.split('_')[0]
            image = misc.imread(os.path.join(base_dir,file))

            names = names.extend(name)
            images.extend(image)

            if(cnt % batch_size == 0):
                images = align_data(mtcnn, images, margin=0)
                embedding_of_batch = get_embedding(sess,input,output,images)

                if(flag):
                    embeddings = embedding_of_batch
                    flag = 0

                cnt = 0

                embeddings = np.concatenate((embedding_of_batch,embeddings))
                images = []

            cnt = cnt +1

    if(cnt != 0):
        images = align_data(mtcnn, images, margin=0)
        embedding_of_batch = get_embedding(sess, input, output, images)

        if (flag):
            embeddings = embedding_of_batch

        embeddings = np.concatenate((embedding_of_batch, embeddings))

    return embeddings,np.array(names)

def load_mtcnn():

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess,os.path.dirname(__file__)+'/trained_params/' )

    return (pnet,rnet,onet)


def align_data(mtcnn, images, margin):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    nrof_samples = len(images)
    img_list = [None] * nrof_samples
    for i in range(nrof_samples):

        img = misc.imread(images[i])
        img_size = np.asarray(img.shape)[0:2]

        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, mtcnn[0], mtcnn[1], mtcnn[2], threshold, factor)
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (160, 160), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

