
from utils.elpv_reader import load_dataset
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os


def countFile(rootDir):

        os.chdir(rootDir) #目录切换到当前目录~
        dirs = os.listdir(rootDir)
        N=0
        for filename in dirs: 
                source_path = rootDir + "\\" + filename
                
                                #源文件路径
                dirsS = os.listdir(source_path)
                source_path = source_path + "\\"
                n = 0
                for file in dirsS:
                        n = n + 1
                print(filename, n)
                N = N + n
        print(N)


images, labels, types, DefectType = load_dataset()

label_names = list(set(DefectType))
number = DefectType.size
typeN = len(label_names)
print('There are %d images in the dataset with %d different types.' %(number,typeN))
print(label_names)
# Divided into train images, test images and validation images
x_train,x_vali,y_train,y_vali= train_test_split(images,DefectType,test_size=0.1,random_state=98)
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size=0.2,random_state=98)
print('\nImages are divided into 3 classes: train images, validation images and test images.')
print('Train images: %d' %y_train.size)
print('Test images: %d' %y_test.size)
print('Validation images: %d' %y_vali.size)


# Save images to their folders.
rootDir_train ="E:\\Application\\elpv-dataset\\newData_r\\train"
rootDir_test ="E:\\Application\\elpv-dataset\\newData_r\\test"
rootDir_vali ="E:\\Application\\elpv-dataset\\newData_r\\vali"
for i in range(0, y_train.size):
        #if y_train[i] == 'Dark-spot':
                DirPath = rootDir_train + '\\' + y_train[i] + '\\' + str(i) + '.png' 
                x_train[i] = cv2.resize(x_train[i], (224, 224), interpolation=cv2.INTER_CUBIC)
                x_train[i]  = cv2.medianBlur(x_train[i] , 3)
                #x_train[i] = SP(x_train[i])
                cv2.imwrite(DirPath, x_train[i])
        

for i in range(0, y_vali.size):
        #if y_vali[i] == 'Dark-spot':
                DirPath = rootDir_vali + '\\' + y_vali[i] + '\\' + str(i) + '.png' 
                x_vali[i] = cv2.resize(x_vali[i], (224, 224), interpolation=cv2.INTER_CUBIC)
                #x_train[i] = SP(x_train[i])
                cv2.imwrite(DirPath, x_vali[i])

for i in range(0, y_test.size):
        #if y_test[i] == 'Dark-spot':
                DirPath = rootDir_test + '\\' + y_test[i] + '\\' + str(i) + '.png'
                x_test[i] = cv2.resize(x_test[i], (224, 224), interpolation=cv2.INTER_CUBIC)
                x_test[i]  = cv2.medianBlur(x_test[i] , 3)

                #x_test[i] = SP(x_test[i])

                cv2.imwrite(DirPath, x_test[i])



countFile(rootDir_train)
coutFile(rootDir_test)
coutFile(rootDir_vali)



'''
import os



rootDir=r"E:\Application\elpv-dataset\test_data\data_original"
saveDir=r"E:\Application\elpv-dataset\test_data\data_nameR"

os.chdir(rootDir) #目录切换到当前目录~
dirs = os.listdir(rootDir)

for filename in dirs: 
        i = 0
        source_path = rootDir + "\\" + filename
        save_path = saveDir + '\\' + filename + '\\' 
        dirsS = os.listdir(source_path)
        n = len(dirsS)
        N = random.sample(range(1,n+1),n)
        print(n,len(N))
        print(filename)
        for file in dirsS:
                read_path = source_path +'\\' + file
                Spath = save_path + 'C'+str(N[i]) + '.png'
                print(read_path,Spath)
                image = cv2.imread(read_path)

                cv2.imwrite(Spath, image)
                i = i + 1


rootDir=r"E:\Application\elpv-dataset\newData_r\vali"
saveDir_train=r"E:\Application\elpv-dataset\newData_dVSf_r\vali\Defective"
os.chdir(rootDir) 
dirs = os.listdir(rootDir)

for filename in dirs: 
        source_path = rootDir + "\\" + filename
        saveP_train = saveDir_train + "\\" + filename 

        dirsS = os.listdir(source_path)
        print(filename)
        if filename != "Functional":
                for file in dirsS:
                        read_path = source_path +'\\' + file                       
                        save_path = saveP_train + file
                        image = cv2.imread(read_path)
                        cv2.imwrite(save_path, image)        
'''





