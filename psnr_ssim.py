from PIL import Image
import cv2
import numpy as np
import math
import glob
import os
def psnr(img1, img2):
    # img1 and img2 have range [0, 255]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def get_gt_predict_pair(path, reblur):
    '''
        input : generated result directory
        output : (gt_path, predict_path) as a zip object
    '''
    predict_folder = path
    predict_filenames = glob.glob(predict_folder + '/*.png')
    predict_filenames.sort(key=lambda x: int(x.split("/")[-1].split('.')[0]))

    if reblur == 'reblur':
        with open('/home/hh/Desktop/disk2/MPGAN/datas/GoPro/test_blur_file.txt', 'r') as f:
            gt_filenames = f.readlines()
        for i, files in enumerate(gt_filenames):
            gt_filenames[i] = files[:-1]
    else:
        with open('/home/hh/Desktop/disk2/MPGAN/datas/GoPro/test_sharp_file.txt', 'r') as f:
            gt_filenames = f.readlines()
        for i, files in enumerate(gt_filenames):
            gt_filenames[i] = files[:-1]

    return zip(gt_filenames,predict_filenames)


def metric_calculator(reblur, path):
    gt_predict_pair = get_gt_predict_pair(path, reblur)

    total_psnr = 0
    total_ssim = 0
    for gt, predict in gt_predict_pair:
        img1 = np.array(Image.open(gt))
        img2 = np.array(Image.open(predict))
        total_psnr += psnr(img1, img2)
        total_ssim += calculate_ssim(img1, img2)

    print(f"{reblur} PSNR = {total_psnr/1111}")
    print(f"{reblur} SSIM = {total_ssim/1111}")
        
if __name__ == '__main__':
    #imglist = ["GOPR0384_11_00"]
    #imgpath = '/home/linnnnn/Research/dataset/GOPRO_Large/test'
    #allFileList = os.path.join(imgpath,imglist[0],"sharp")
    #for filename in os.listdir(allFileList):
        #print(filename)
    img1=Image.open('/home/hh/Desktop/disk2/MPGAN/runs/reblur_DMPHN_1_2_4_c32/result_picture/deblur/99.png')
    img2=Image.open('/home/hh/Desktop/GoPro/test/GOPR0384_11_00/sharp/000100.png')
    i1_array = np.array(img1)
    i2_array = np.array(img2)

    #calculate_metric()
