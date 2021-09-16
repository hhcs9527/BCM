import os
import json

def add_path(train):
    root = f'/home/hh/Desktop/disk2/gopro/{train}'

    traversal_directories = sorted(os.listdir(root))

    train_list = []

    for directory in traversal_directories:
        current_dir = os.path.join(root, directory)
        for files in os.listdir(current_dir):
            train_list.append(os.path.join(current_dir,files))

    to_place = '/home/hh/Desktop/disk2/MPGAN/datas/GoPro/one_all'        
    f = open(f"{to_place}/{train}_sharp.txt", "w")
    for i in range(len(train_list)-1):
        f.write(train_list[i] + '\n')
    f.write(train_list[i])
    f.close()

def read_all_path(train):
    root = f'/home/hh/Desktop/disk2/gopro/{train}'

    traversal_directories = sorted(os.listdir(root))

    train_list = []

    for directory in traversal_directories:
        current_dir = os.path.join(root, directory)
        for files in os.listdir(current_dir):
            train_list.append(os.path.join(current_dir,files))

    return sorted(train_list)

def find_pairs(train):
    '''
        save blur -> K sharp images pairs to json files

        output_file:
            {"/home/hh/Desktop/GoPro/train/GOPR0372_07_00/blur/000047.png":
                    ["/home/hh/Desktop/disk2/gopro/train/GOPR0372_07_00/000323.png"
                    ,"/home/hh/Desktop/disk2/gopro/train/GOPR0372_07_00/000324.png"
                    ,"/home/hh/Desktop/disk2/gopro/train/GOPR0372_07_00/000325.png"
                    ,"/home/hh/Desktop/disk2/gopro/train/GOPR0372_07_00/000326.png"
                    ,"/home/hh/Desktop/disk2/gopro/train/GOPR0372_07_00/000327.png"
                    ,"/home/hh/Desktop/disk2/gopro/train/GOPR0372_07_00/000328.png"
                    ,"/home/hh/Desktop/disk2/gopro/train/GOPR0372_07_00/000329.png"]...}

        file name :            
            train/test_pairs.json
    '''

    root = f'/home/hh/Desktop/disk2/MPGAN/datas/GoPro/one_all/{train}_blur_file.txt' 
    f = open(root, 'r')
    train_pairs = {}

    blur_list = f.read().split('\n')[:-1]
    sharp_list = read_all_path(train)
    sharp_index = 0
    for blur_path in blur_list:
        num_shaprs = int(blur_path.split('/')[-3].split('_')[-2])
        compose = sharp_list[sharp_index : sharp_index + num_shaprs]
        train_pairs[blur_path] = compose
        sharp_index += num_shaprs
    
    with open(f'/home/hh/Desktop/disk2/MPGAN/datas/GoPro/one_all/{train}_pairs.json', 'w') as json_file:
        json.dump(train_pairs, json_file)
    

if __name__ == "__main__":   
    find_pairs('train')
    find_pairs('test')
    import cv2
    import numpy as np

    root = f'/home/hh/Desktop/disk2/MPGAN/datas/GoPro/one_all/train_sharp_file.txt' 
    f = open(root, 'r')
    blur_list = f.read().split('\n')[:-1]
    root = f'/home/hh/Desktop/disk2/MPGAN/datas/GoPro/one_all/train_blur_file.txt' 
    f = open(root, 'r')
    sharp_list = f.read().split('\n')[:-1]

    origin_pair = zip(sharp_list, blur_list)

    with open('/home/hh/Desktop/disk2/MPGAN/datas/GoPro/one_all/train_pairs.json') as json_file:
        data = json.load(json_file)

    print(data['/home/hh/Desktop/GoPro/train/GOPR0871_11_01/blur/000280.png'])
    '''

    for b, s in origin_pair:
        length = len(data[b])
        sharp = cv2.imread(s) 
        blur = cv2.imread(b) 

        res = []
        zero = False
        index = 0
        for i in range(length):
            all_sharp = cv2.imread(data[b][i])
            result = np.sum(sharp-all_sharp)
            res.append(result)
            if not result:
                zero = True
                index = i

        if index != length//2:
            print(index, b, length)
    '''