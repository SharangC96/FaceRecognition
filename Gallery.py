import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import util
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from scipy import misc


class Gallery:

    sess = tf.Session()
    input, embedding, phase_train = util.load_model(sess)
    mtcnn = util.load_mtcnn()

    def __init__(self, gallery_dir,name='MyGallery',reuse = False):

        self.path = os.path.join(os.path.dirname(__file__),'trained_params',name)
        self.clf = None
        self.encoder = None

        if(not reuse):
            self.clf = KNeighborsClassifier(n_neighbors=1, weights='distance', p = 2, n_jobs= -1)

            x, y = util.create_dataset(self.sess, self.mtcnn, self.input, self.embedding, gallery_dir,self.phase_train)
            print('Dataset Created\n','Input to KNN (X,y):',x.shape,' ',y.shape)

            self.encoder = LabelEncoder()
            self.encoder.fit(y)
            y = self.encoder.transform(y)

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

    def recognize(self, image_path, threshold=0.9):

        image = misc.imread(image_path)
        image = util.align_data(self.mtcnn,[image],0)
        this_embedding = util.get_embedding(self.sess,self.input, self.embedding, image,self.phase_train)

        a,_ = self.clf.kneighbors(X=this_embedding,n_neighbors= 1)
        a = a.reshape(-1)

        name = self.encoder.inverse_transform(self.clf.predict(this_embedding))
        name[ a > threshold] = 'Cannot Recognize'

        return name