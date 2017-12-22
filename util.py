import tensorflow as tf
from InceptionResNetv2 import inception_resnet_v2
import numpy as np

def load_model(sess,path):

    x = tf.placeholder(shape=[None,299,299,3],dtype=tf.float32,name='input')
    out = inception_resnet_v2(x)

    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

    return x,out

def get_embedding(sess, input, output, image):

    embedding = sess.run(output,feed_dict={input:image})
    if(image.shape[0]==1):
        return embedding.reshape(128)

    return embedding

def create_dataset(sess, input, output ,base_dir):

    #read files from the directory that is passed
    filename_queue = tf.train.string_input_producer(base_dir,num_epochs=1,capacity=100)

    image_reader = tf.WholeFileReader()
    name_file,image_file = image_reader.read(filename_queue)

    #decode image
    image_file = tf.image.decode_jpeg(image_file,channels=3)

    #start the queues
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    #get first batch of the labels and images
    em = get_embedding(sess, input, output, image_file)

    try:
        while not coord.should_stop():

            # get images for other batches
            em_temp = get_embedding(sess, input, output, image_file)

            # concatenate the images
            y = np.concatenate((y, name_file), axis=0)
            em = np.concatenate((em, em_temp), axis=0)


    except tf.errors.OutOfRangeError:
        print('Dataset Ready')

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

    return (em,y)

create_dataset('/home/sharang/Pictures/Webcam')



