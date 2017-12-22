import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from util import create_dataset, get_embedding,load_model
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

class Gallery:

    def __init__(self, gallery_dir,data_dir_path,name='MyGallery',reuse = False):

        self.path = os.path.join(data_dir_path, name)
        self.clf = None
        self.encoder = None
        self.sess = tf.Session()

        self.input, self.embedding = load_model(self.sess,self.path)

        if(not reuse):
            self.clf = KNeighborsClassifier(n_neighbors=1, weights='distance', p = 2, n_jobs= -1)

            x, y = create_dataset(self.sess, self.input, self.embedding, gallery_dir)

            self.encoder = LabelEncoder()
            self.encoder.fit(y)
            y = self.encoder.transform(y)

            self.clf.fit(x, y)

            joblib.dump(self.encoder, self.path+'.pkl')
            joblib.dump(self.clf, self.path+'.pkl')

        else:
            clf_path = os.path.join(data_dir_path, 'clf')
            self.clf = joblib.load(clf_path)

            enc_path = os.path.join(data_dir_path, 'enc')
            self.encoder = joblib.load(enc_path)


    def recognize(self, image):

        this_embedding = get_embedding(self.sess, self.input,self.embedding,image)
        name = self.encoder.inverse_transform(self.clf.predict(this_embedding))

        return name