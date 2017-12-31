from Gallery import Gallery
import os

dirname = os.path.dirname(__file__)
gallery_dir = os.path.join(dirname,'Datasets','train_data_4')

testing = Gallery(gallery_dir,name='testing',reuse= False,use_lmnn= False)

print(testing.accuracy_dir(os.path.join(dirname,'Datasets','test_data_4')))

# print(mygallery.recognize_image(os.path.join(dirname,'test.jpg')))
