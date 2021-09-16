from calculate_metric import PSNR, SSIM
from PIL import Image
import torch
from torchvision import transforms, datasets
import json, cv2, os
import numpy as np

experiment = 'SOTA'
Compare = ['BRRMv5_DMPHN_1_2_4_c32', 'BRRM_DMPHN_1_2_4_c32', 'BRRML_DMPHN_1_2_4_c32', 'DMPHN']
Compare = ['MPRNet_BRMv2', 'MPRNet', 'BRRMv5_DMPHN_1_2_4_c32', 'BRRMv5_DMPHN_1_2_4_8_c32']

def read(file, i):
    compare_path = f'/home/hh/Desktop/disk2/MPGAN/runs/{file}/result_picture/deblur/{i}.png'
    restore = Image.open(compare_path)
    return restore

def read_file(file, i):
    compare_path = f'/home/hh/Desktop/disk2/MPGAN/runs/{file}/result_picture/deblur/{i}.png'
    gt_path = f'/home/hh/Desktop/disk2/MPGAN/runs/Sharp/result_picture/deblur/{i}.png'
    restore = transforms.ToTensor()(Image.open(compare_path).convert('RGB'))
    gt = transforms.ToTensor()(Image.open(gt_path).convert('RGB'))
    return PSNR(gt, restore)


def valid(arr):
    if max(arr) == arr[0]: return True
    return False

def compare(num = 1111):
    Base = ['Sharp']
    result = []
    for i in range(num):
        tmp = []
        for comp in Compare:
            tmp.append(read_file(comp, i))
        result.append(tmp)
    return result

def turn2json(result):
    valid_seq = {}
    for i, res in enumerate(result):
        if valid(res): 
            valid_img = []
            valid_seq[i] = res
            for comp in Compare: valid_img.append(read(comp, i))
            val = Image.fromarray(np.uint8(np.vstack(valid_img)))
            val.save(f'./compare_result/{experiment}/comp_{i}.png')

    with open(f'./compare_result/{experiment}/choose_{experiment}.json', 'w') as outfile:
        json.dump(valid_seq, outfile)

def find_match():
    turn2json(compare())

def generate_comb():
    f = open(f'./compare_result/{experiment}/choose_{experiment}.json')
    data = json.load(f) 
    for i in data.keys():
        valid_img = []
        for comp in Compare: valid_img.append(read(comp, i))
        val = Image.fromarray(np.uint8(np.vstack(valid_img)))
        val.save(f'./compare_result/{experiment}/comp_{i}.png')
    f.close()
    


def copy_file(id, color):

    def opencvRead(exp, id):
        return cv2.imread(f'/home/hh/Desktop/disk2/MPGAN/runs/{exp}/result_picture/deblur/{id}.png')
    
    # check if there exist a dir for crop/box result
    box = f'./compare_result/{experiment}/MPR/p{id}/box/'
    crop = f'./compare_result/{experiment}/MPR/p{id}/crop/'
    if not os.path.isdir(box): os.mkdir(box)
    if not os.path.isdir(crop): os.mkdir(crop)

    # iterate all the compare & mark them + crop them, and save
    for exp in Compare:
        mv = opencvRead(exp, id)
        img_box, img_crop = CropAndBox(exp, id, color)
        cv2.imwrite(f'{box}{exp}.png', img_box)
        cv2.imwrite(f'{crop}{exp}.png', img_crop)


def CropAndBox(file, id, color):
    lookup = {'Y':(0, 255, 255), 'G':(0, 255, 0), 'R': (0, 0, 255), 'B': (255, 0, 0), 'LB': (255, 255, 0)}
    f = open(f'./compare_result/{experiment}/MPR/p{id}/p{id}.json')
    data = json.load(f)
    f.close()
    img = cv2.imread(f'/home/hh/Desktop/disk2/MPGAN/runs/{file}/result_picture/deblur/{id}.png')

    x, y = data['x'], data['y']
    dx, dy = data['dx'], data['dy']

    img_crop = img[y:y+dy, x:x+dx]
    img_box = img.copy()
    img_box = cv2.rectangle(img_box, (x, y), (x+dx, y+dy), lookup[color], 2)

    return img_box, img_crop



def copy_org(id, color):

    def opencvRead(file, id):
        return cv2.imread(f'/home/hh/Desktop/disk2/MPGAN/runs/{file}/result_picture/deblur/{id}.png')

    box = f'./compare_result/{experiment}/MPR/p{id}/box/'
    crop = f'./compare_result/{experiment}/MPR/p{id}/crop/'

    # iterate all the compare & mark them + crop them, and save
    for exp in ['reblur', 'deblur']:
        mv = opencvRead(exp, id)
        img_box, img_crop = CropAndBox_org(exp, id, color)
        cv2.imwrite(f'{box}{exp}.png', img_box)
        cv2.imwrite(f'{crop}{exp}.png', img_crop)


def CropAndBox_org(file, id, color):
    lookup = {'Y':(0, 255, 255), 'G':(0, 255, 0), 'R': (0, 0, 255), 'B': (255, 0, 0), 'LB': (255, 255, 0)}
    f = open(f'./compare_result/{experiment}/MPR/p{id}/p{id}.json')
    data = json.load(f)
    f.close()
    img = cv2.imread(f'/home/hh/Desktop/disk2/MPGAN/runs/Sharp/result_picture/{file}/{id}.png')
    x, y = data['x'], data['y']
    dx, dy = data['dx'], data['dy']
    
    img_crop = img[y:y+dy, x:x+dx]
    img_box = img.copy()
    img_box = cv2.rectangle(img_box, (x, y), (x+dx, y+dy), lookup[color], 2)

    return img_box, img_crop

def CROPall(id, color):
    copy_file(id, color)
    copy_org(id, color)    


def crop_hands(location = 0, size = '', file = ''):
    path = f'./compare_result/{experiment}/ablation/normal/'
    for p in os.listdir(path):
        img = cv2.imread(f'{path}{p}')
        x, y = 470, 407
        dx, dy = 65, 60
        crop_img = img[y:y+60, x:x+65]
        cv2.imwrite(f'./compare_result/{experiment}/ablation/crop/{p}', crop_img)

def crop_test(id):
    def opencvRead(id, exp = 'MPRNet_BRMv2'):
        return cv2.imread(f'/home/hh/Desktop/disk2/MPGAN/runs/{exp}/result_picture/deblur/{id}.png')
    img = opencvRead(id)
    #cv2.imshow('window', img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    x, y = 88, 401
    dx, dy = 180, 60
    crop_img = img[y:y+dy, x:x+dx]
    cv2.imwrite(f'crop_test.png', crop_img)

def crop_coord(id):
    def opencvRead(id, exp = 'MPRNet_BRMv2'):
        return cv2.imread(f'/home/hh/Desktop/disk2/MPGAN/runs/{exp}/result_picture/deblur/{id}.png')
    img = opencvRead(id)
    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('generating')
    #crop_hands()
    #find_match()
    #generate_comb()
    for id in [2, 103, 104, 205, 541, 519]:
        CROPall(id, 'Y')
    #crop_coord(103)
    #crop_test(541)
    


    



    