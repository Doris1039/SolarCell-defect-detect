import csv   #导入csv模块
import cv2
from PIL import Image
import numpy as np
import os




def load_dataset(fname=None):
    if fname is None:
        # Assume we are in the utils folder and get the absolute path to the
        # parent directory.
        fname = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir))
        fname = os.path.join(fname, "E:\\Application\\elpv-dataset\\data_test\\labels.csv")

    data = np.genfromtxt(fname, delimiter=' ', dtype=['|S100', '|S20'], names=[
                         'path', 'DefectType'])
    image_fnames = np.char.decode(data['path'])
    #image_name = np.char.decode(data['path'])

    DefectType = np.char.decode(data['DefectType'])

    def load_cell_image(fname):
        with Image.open(fname) as image:
            return np.asarray(image)

    dir = os.path.dirname(fname)

    images = np.array([load_cell_image(os.path.join(dir, fn))
                       for fn in image_fnames])

    return images, DefectType

def AddArray(rootDir,Path,Type,Num):
    os.chdir(rootDir) #目录切换到当前目录~
    dirs = os.listdir(rootDir)
    class_names = ['Defective', 'Functional']

    for filename in dirs: 
        source_path = rootDir + "\\" + filename
        dirsS = os.listdir(source_path)
        for file in dirsS:
            Spath = source_path + '\\' + file
            Path.append(Spath)
            Type.append(filename)
            for i in range(2):
                if filename == class_names[i]:
                    Num.append(i)

    return Path, Type, Num

Path = []
Type = []
Num = []

rootDir_Test = r"E:\Application\elpv-dataset\newData_dVSf_r\test"
rootDir_Train = r"E:\Application\elpv-dataset\test\train"

Path, Type, Num = AddArray(rootDir_Test,Path,Type,Num)
#Path, Type, Num = AddArray(rootDir_Train,Path,Type,Num)



print(len(Path))
print(len(Type))
print(len(Num))


np.savetxt(r"E:\Application\elpv-dataset\data_test\labels_eval_dn.csv", np.column_stack((Path,Type,Num)),fmt='%s')

'''
images,  DefectType, Num = load_dataset()


label_names = list(set(DefectType))
number = DefectType.size

print(DefectType)
print('There are %d images in the dataset with %d different types.' %(number,number))


cv2.imshow('lena3',images[0])
cv2.waitKey(0)


with open(r"E:\Application\elpv-dataset\data_test\dd.csv",'w',newline='',encoding = 'utf-8') as f:
    y_test = [1,2,'dd']
    y_train = ['df', 'dfs']
    writer = csv.writer(f)
    writer.writerows((y_test,y_train))
'''