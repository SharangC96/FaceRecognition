import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import util
import tensorflow as tf
from scipy import misc
import numpy as np
from metric_learn import LMNN

class Gallery:

    sess = tf.Session()
    input, embedding, phase_train = util.load_model(sess)
    mtcnn = util.load_mtcnn()

    def __init__(self, gallery_dir,name='MyGallery',reuse = False,use_lmnn = False):

        self.path = os.path.join(os.path.dirname(__file__),'trained_params',name)
        self.clf = None
        self.encoder = None
        self.use_lmnn = use_lmnn
        self.lmnn = None

        if(not reuse):

            self.clf = KNeighborsClassifier(n_neighbors=3, weights='distance', p = 2, n_jobs= -1)

            x, y = util.create_dataset(self.sess, self.mtcnn, self.input, self.embedding, gallery_dir,self.phase_train)
            print('Dataset Created\n','Input to KNN (X,y):',x.shape,' ',y.shape)

            self.encoder = LabelEncoder()
            self.encoder.fit(y)

            y = self.encoder.transform(y)

            if(use_lmnn):
                self.lmnn = LMNN(k=3, max_iter=350, verbose=True, learn_rate=0.00002)
                self.lmnn.fit(x,y)
                joblib.dump(self.lmnn, os.path.join(self.path, 'lmnn.joblib.pkl'))
                x = self.lmnn.transform(x)

            self.clf.fit(x, y)

            if not os.path.exists(self.path):
                os.makedirs(self.path)

            joblib.dump(self.encoder, os.path.join(self.path,'enc.joblib.pkl'))
            joblib.dump(self.clf, os.path.join(self.path,'clf.joblib.pkl'))


        else:
            clf_path =  os.path.join(self.path,'clf.joblib.pkl')
            self.clf = joblib.load(clf_path)

            enc_path =  os.path.join(self.path,'enc.joblib.pkl')
            self.encoder = joblib.load(enc_path)

            if(use_lmnn):
                lmnn_path = os.path.join(self.path,'lmnn.joblib.pkl')
                self.lmnn = joblib.load(lmnn_path)

    def recognize_image(self, image_path):

        image = misc.imread(image_path)
        image = util.align_data(self.mtcnn,[image],0)
        this_embedding = util.get_embedding(self.sess,self.input, self.embedding, image,self.phase_train)

        if(self.use_lmnn):
            this_embedding = self.lmnn.transform(this_embedding)

        name = self.encoder.inverse_transform(self.clf.predict(this_embedding))

        return name

    def accuracy_dir(self,dir_path):

        x, y = util.create_dataset(self.sess, self.mtcnn, self.input, self.embedding, dir_path, self.phase_train)

        if(self.use_lmnn):
            x = self.lmnn.transform(x)

        name = self.encoder.inverse_transform(self.clf.predict(x))

        len = name.shape[0]
        cnt1 = np.sum(name == y)    #correct classifications
        cnt2 = len - cnt1           #misclassification

        return cnt1,cnt2