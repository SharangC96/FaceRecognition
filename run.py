from Gallery import Gallery
import os

dirname = os.path.dirname(__file__)
gallery_dir = os.path.join(dirname,'Dataset')

#gallery_dir = 'Dataset/'
mygallery = Gallery(gallery_dir,name='MyGallery',reuse= False)

print(mygallery.recognize(os.path.join(dirname,'test.jpg')))
# print(mygallery.recognize('test.jpg'))