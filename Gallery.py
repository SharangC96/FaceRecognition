import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import util
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from scipy import misc
class Gallery:

    sess = tf.Session()
    input, embedding = util.load_model(sess)
    mtcnn = util.load_mtcnn()

    def __init__(self, gallery_dir,name='MyGallery',reuse = False):

        self.path = os.path.join(os.path.dirname(__file__),'trained_params',name)
        self.clf = None
        self.encoder = None

        if(not reuse):
            self.clf = KNeighborsClassifier(n_neighbors=1, weights='distance', p = 2, n_jobs= -1)

            x, y = util.create_dataset(self.sess, self.mtcnn, self.input, self.embedding, gallery_dir)

            self.encoder = LabelEncoder()
            self.encoder.fit(y)
            y = self.encoder.transform(y)

            self.clf.fit(x, y)

            joblib.dump(self.encoder, os.path.join(self.path,'enc.pkl'))
            joblib.dump(self.clf, os.path.join(self.path,'clf.pkl'))

        else:
            clf_path =  os.path.join(self.path,'clf.pkl')
            self.clf = joblib.load(clf_path)

            enc_path =  os.path.join(self.path,'enc.pkl')
            self.encoder = joblib.load(enc_path)


    def recognize(self, image_path):

        image = misc.imread(image_path)
        util.align_data(self.mtcnn,[image],0)
        this_embedding = util.get_embedding(self.sess,self.input, self.embedding, image)
        name = self.encoder.inverse_transform(self.clf.predict(this_embedding))

        return name