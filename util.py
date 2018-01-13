import tensorflow as tf
import numpy as np
import os
import align.detect_face
from scipy import misc
import pandas as pd


def load_model(sess):
    dirname = os.path.dirname(__file__)

    saver = tf.train.import_meta_graph(dirname + '/trained_params/model-20170512-110547.meta',
                                       clear_devices=True)  # load graph
    saver.restore(sess, dirname + '/trained_params' + '/model-20170512-110547.ckpt-250000')  # load weights

    x = tf.get_default_graph().get_tensor_by_name('input:0')  # input placeholder
    out = tf.get_default_graph().get_tensor_by_name("embeddings:0")  # output tensor
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    return x, out, phase_train_placeholder


def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def get_embedding(sess, input, output, image, phase_train_placeholder):
    embedding = sess.run(output, feed_dict={input: image, phase_train_placeholder: False})
    return embedding

def create_dataset(sess, mtcnn, input, output, base_dir, phase_train_placeholder):

    if(os.path.exists(base_dir+'/data.csv')):
        x = pd.read_csv(base_dir+'/data.csv')
        y = x['name'].values
        x = x.iloc[:,1:-1].values

        return x,y.tolist()

    batch_size = 300

    embeddings = None
    cnt = 0
    flag = 1
    total_batches = 0
    names = []
    images = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:

            name = root.split('/')[-1]
            try:
                image = np.array(misc.imread(os.path.join(root,file)))
            except:
                print('Cannot read .csv as image file!')

            if(len(image.shape)<3):
                continue

            names.append(name)
            images.append(image)
            cnt = cnt + 1

            if(cnt % batch_size == 0):
                images = align_data(mtcnn, images, margin=0)
                embedding_of_batch = get_embedding(sess,input,output,images, phase_train_placeholder)

                if(flag):
                    embeddings = embedding_of_batch
                    flag = 0

                else:
                    embeddings = np.concatenate((embeddings,embedding_of_batch))

                cnt = 0
                images = []

                print(total_batches+1,'Batch Processed')
                total_batches  = total_batches +1


    if(cnt != 0):
        images = align_data(mtcnn, images, margin=0)
        print('last batch- ',images.shape)
        embedding_of_batch = get_embedding(sess, input, output, images, phase_train_placeholder)

        if (flag):
            embeddings = embedding_of_batch
        else:
            embeddings = np.concatenate((embeddings,embedding_of_batch))

    table = pd.DataFrame(embeddings)
    table['name'] = names
    table.to_csv(base_dir+'/data.csv')

    return embeddings,names


def load_mtcnn():
    print('Creating networks and loading parameters MTCCN')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, os.path.dirname(__file__) + '/trained_params/')

    return pnet, rnet, onet


def align_data(mtcnn, images, margin):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    nrof_samples = len(images)
    for i in range(nrof_samples):
        img = images[i]

        img_size = np.asarray(img.shape)[0:2]
        img_center = np.array([img.shape[0]/2,img.shape[1]/2])


        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, mtcnn[0], mtcnn[1], mtcnn[2], threshold, factor)
        nrof_faces = bounding_boxes.shape[0]

        if(nrof_faces == 0):
            aligned = misc.imresize(img, (160, 160), interp='bilinear')
            prewhitened = prewhiten(aligned)
            images[i] = prewhitened
            continue

        correct_face = np.zeros(nrof_faces)

        for j in range(nrof_faces):
            det = np.squeeze(bounding_boxes[j, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            crop_center = np.array([(bb[1]+bb[3])/2 , (bb[0]+bb[2])/2])
            area = (bb[3]-bb[1])*(bb[2]-bb[0])
            dist = np.sum(np.square(img_center-crop_center))
            correct_face[j] = dist

        correct_face = np.argmin(correct_face,axis=0)

        det = np.squeeze(bounding_boxes[correct_face, 0:4])

        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = misc.imresize(cropped, (160, 160), interp='bilinear')
        prewhitened = prewhiten(aligned)
        images[i] = prewhitened

    if nrof_samples == 1:
        return np.reshape(images[0],newshape=[1,160,160,3])
    else:
        images = np.stack(images)

    return images

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y
