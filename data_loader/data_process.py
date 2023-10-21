
import os
import random
import shutil

'''

occluded_dataset_path = os.path.join('G:/datasets/occluded_data/Occluded_REID_OURS', 'occluded_body_images/')
whole_dataset_path = os.path.join('G:/datasets/occluded_data/Occluded_REID_OURS', 'whole_body_images/')
new_dataset_path = 'G:\datasets\occluded_data\Occluded_REID_OURS/new'
for files in os.listdir(occluded_dataset_path):
    for img in os.listdir(os.path.join(occluded_dataset_path, files)):
        new_name = img.split('.')[0]+'_01'+'.tif'
        shutil.copy(os.path.join(occluded_dataset_path, files, img), os.path.join(new_dataset_path, new_name))

for files in os.listdir(whole_dataset_path):
    for img in os.listdir(os.path.join(whole_dataset_path, files)):
        new_name = img.split('.')[0]+'_02'+'.tif'
        shutil.copy(os.path.join(whole_dataset_path, files, img), os.path.join(new_dataset_path, new_name))
        
dataset_path = 'G:\datasets\occluded_data\Occluded_REID_OURS/new'
imgs = os.listdir(dataset_path)
for i in range(10):
    new_dataset_path = 'G:\datasets\occluded_data\Occluded_REID_OURS/new'+str(i)
    pids = list(range(1, 201))
    train_pids = random.sample(pids, int(len(pids) / 2))
    os.makedirs(os.path.join('G:\datasets\occluded_data\Occluded_REID_OURS/new'+str(i), 'bounding_box_train'))
    os.makedirs(os.path.join('G:\datasets\occluded_data\Occluded_REID_OURS/new' + str(i), 'query'))
    os.makedirs(os.path.join('G:\datasets\occluded_data\Occluded_REID_OURS/new' + str(i), 'bounding_box_test'))
    for img in imgs:
        pid, cid = img.split('_')[0], img.split('_')[2].split('.')[0]
        if int(pid) in train_pids:
            shutil.copy(os.path.join(dataset_path, img), os.path.join(new_dataset_path, 'bounding_box_train', img))
        else:
            if int(cid) == 1:
                shutil.copy(os.path.join(dataset_path, img), os.path.join(new_dataset_path, 'query', img))
            else:
                shutil.copy(os.path.join(dataset_path, img), os.path.join(new_dataset_path, 'bounding_box_test', img))
                
            
                '''

pduke_train_occluded_dataset_path = os.path.join('G:/datasets/occluded_data/P_Duke_OURS/train', 'occluded_body_images/')
pduke_train_whole_dataset_path = os.path.join('G:/datasets/occluded_data/P_Duke_OURS/train', 'whole_body_images/')
pduke_test_occluded_dataset_path = os.path.join('G:/datasets/occluded_data/P_Duke_OURS/test', 'occluded_body_images/')
pduke_test_whole_dataset_path = os.path.join('G:/datasets/occluded_data/P_Duke_OURS/test', 'whole_body_images/')
new_dataset_path = 'G:\datasets\occluded_data\P_duke_OURS/new'
os.makedirs(os.path.join(new_dataset_path, 'bounding_box_train'))
os.makedirs(os.path.join(new_dataset_path, 'query'))
os.makedirs(os.path.join(new_dataset_path, 'bounding_box_test'))
for files in os.listdir(pduke_train_occluded_dataset_path):
    for img in os.listdir(os.path.join(pduke_train_occluded_dataset_path, files)):
        new_name = img.split('.')[0]+'_01'+'.jpg'
        shutil.copy(os.path.join(pduke_train_occluded_dataset_path, files, img), os.path.join(new_dataset_path,
                                                    'bounding_box_train', new_name))

for files in os.listdir(pduke_train_whole_dataset_path):
    for img in os.listdir(os.path.join(pduke_train_whole_dataset_path, files)):
        new_name = img.split('.')[0]+'_02'+'.jpg'
        shutil.copy(os.path.join(pduke_train_whole_dataset_path, files, img), os.path.join(new_dataset_path,
                                                    'bounding_box_train', new_name))

for files in os.listdir(pduke_test_occluded_dataset_path):
    for img in os.listdir(os.path.join(pduke_test_occluded_dataset_path, files)):
        new_name = img.split('.')[0]+'_01'+'.jpg'
        shutil.copy(os.path.join(pduke_test_occluded_dataset_path, files, img), os.path.join(new_dataset_path,
                                                    'query', new_name))

for files in os.listdir(pduke_test_whole_dataset_path):
    for img in os.listdir(os.path.join(pduke_test_whole_dataset_path, files)):
        new_name = img.split('.')[0]+'_02'+'.jpg'
        shutil.copy(os.path.join(pduke_test_whole_dataset_path, files, img), os.path.join(new_dataset_path,
                                                    'bounding_box_test', new_name))


