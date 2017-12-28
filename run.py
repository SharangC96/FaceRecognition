from Gallery import Gallery

gallery_dir = '/home/sharang/Pictures/Dataset'
mygallery = Gallery(gallery_dir,name='MyGallery',reuse= False)

print(mygallery.recognize('/home/sharang/Pictures/test.jpg'))