import os
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import util
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Gallery:

    sess = tf.Session()
    input, embedding, phase_train = util.load_model(sess)
    mtcnn = util.load_mtcnn()

    def __init__(self, gallery_dir,name='MyGallery',reuse = False,r=1,p=2):

        self.path = os.path.join(os.path.dirname(__file__),'trained_params',name)
        self.clf = None
        self.encoder = None
        self.lmnn = None

        if(not reuse):
            x, y = util.create_dataset(self.sess, self.mtcnn, self.input, self.embedding, gallery_dir, self.phase_train)
            print('Dataset Created\n', 'Input to RNN (X,y):', x.shape)

            y_copy = y.copy()
            y_copy.append('Cannot Recognize')

            self.encoder = LabelEncoder()
            self.encoder.fit(y_copy)

            outlier_label = self.encoder.transform(['Cannot Recognize'])
            self.clf = RadiusNeighborsClassifier(radius=r,weights='distance',outlier_label=outlier_label,p=p)

            self.clf.fit(x, self.encoder.transform(y))

            if not os.path.exists(self.path):
                os.makedirs(self.path)

            joblib.dump(self.encoder, os.path.join(self.path,'enc.joblib.pkl'))
            joblib.dump(self.clf, os.path.join(self.path,'clf.joblib.pkl'))


        else:
            clf_path =  os.path.join(self.path,'clf.joblib.pkl')
            self.clf = joblib.load(clf_path)

            enc_path =  os.path.join(self.path,'enc.joblib.pkl')
            self.encoder = joblib.load(enc_path)

    def recognize_image(self, image_path):

        image = plt.imread(image_path)
        image = util.align_data(self.mtcnn,[image],10)
        this_embedding = util.get_embedding(self.sess,self.input, self.embedding, image,self.phase_train)

        prediction = self.clf.predict(this_embedding)
        name = self.encoder.inverse_transform(prediction)

        return name

    def accuracy_dir(self,dir_path):

        x, y = util.create_dataset(self.sess, self.mtcnn, self.input, self.embedding, dir_path, self.phase_train)

        predicted_labels = self.clf.predict(x)
        predicted_names = self.encoder.inverse_transform(predicted_labels)
        acc = np.mean((predicted_names == y).astype(int))
        cannot = np.mean((predicted_names == 'Cannot Recognize').astype(int))
        far = 1 - acc - cannot

        results ={'VAL': np.mean(acc)*100,
                  'FAR':far*100,
                  'Cannot Recognize': np.mean(cannot)*100}

        return results