import os
from shutil import copyfile

train_examples = 4
base_dir = os.path.dirname(__file__)

save_path = os.path.join(base_dir,'Datasets','train_data_ex'+'_'+str(train_examples))
save_path_2 = os.path.join(base_dir,'Datasets','test_data_ex'+'_'+str(train_examples))

cnt = 0
cnt2 = 0
for a, b, c in os.walk(os.path.join(base_dir,'Datasets','lfw')):
    folder_name = a.split('/')[-1]
    if (len(c) > train_examples):

        cnt = cnt + 1
        dest = os.path.join(save_path, folder_name)

        if not os.path.exists(dest):
            os.makedirs(dest)

        for i in range(train_examples):
            copyfile(os.path.join(a,c[i]), os.path.join(dest,c[i]))

    else:
        cnt2 = cnt2 +1
        dest2 = os.path.join(save_path_2, folder_name)

        if not os.path.exists(dest2):
            os.makedirs(dest2)

        for i in c:
            copyfile(os.path.join(a, i), os.path.join(dest2, i))

print('No of unique faces:', cnt,cnt2)