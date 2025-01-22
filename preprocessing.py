import os
import shutil

''' Create the folders '''
if not os.path.exists('train'):
    os.makedirs('train')
    os.makedirs('train/0')
    os.makedirs('train/1')
    os.makedirs('train/2')
    os.makedirs('train/3')

if not os.path.exists('test'):
    os.makedirs('test')
    os.makedirs('test/0')
    os.makedirs('test/1')
    os.makedirs('test/2')
    os.makedirs('test/3')

TRAIN_DIR = 'train'
TEST_DIR = 'test'
CLASSES_DIR = 'Test_4cl_amostra/'
patient_ids = []
patient_added = []

for dir in os.listdir(CLASSES_DIR):

    ''' Get all patient ids '''
    patient_ids = []
    for file in os.listdir(CLASSES_DIR + dir):
        patient_id = file.split('_')[0]
        technique_id = file.split('_')[1]

        if patient_id != '':
            patient_ids.append(patient_id)

    ''' Get the number of different patients '''
    len_patient = len(set(patient_ids))
    different_patients = set(patient_ids)

    patient_added = []
    min_patients = -(-len_patient * 60 // 100)

    cont = 1

    ''' Get all images from the same patient '''
    for patient in different_patients:

        if cont <= min_patients:
            for file in os.listdir(CLASSES_DIR + dir):
                patient_id = file.split('_')[0]
                technique_id = file.split('_')[1]

                if patient_id == patient:
                    #print(f"I will move {file} to {TRAIN_DIR}/{dir}")
                    shutil.copy(CLASSES_DIR + dir + '/' + file, TRAIN_DIR + '/' + dir)
        else:
            for file in os.listdir(CLASSES_DIR + dir):
                patient_id = file.split('_')[0]
                technique_id = file.split('_')[1]

                if patient_id == patient:
                    #print(f"I will move {file} to {TEST_DIR}/{dir}")
                    shutil.copy(CLASSES_DIR + dir + '/' + file, TEST_DIR + '/' + dir)
    
        cont+=1
        
        