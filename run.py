from Gallery import Gallery

file_no = 2
gallery_dir = '/home/sharang/Documents/ML/FaceRecognition/Datasets/train_data_'+str(file_no)
mygallery = Gallery(gallery_dir,name='testing_4',reuse= False,p=2,r=0.905)

mygallery.accuracy_dir('/home/sharang/Documents/ML/FaceRecognition/Datasets/test_data_'+str(file_no),threshold=0.60)