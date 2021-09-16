close all;clear all;
datasets = {'GoPro'};
% read path first
fileID = fopen('./datas/GoPro/test_sharp_file.txt','r');
target_file = textscan(fileID, '%s','delimiter','\n'); 
fclose(fileID);    
total_psnr = 0;
total_ssim = 0;
experiment = 'BRRMv5ALL_DMPHN_1_2_4_c32';
for j = 1:1111
    input = imread(strcat('./runs/',experiment,'/result_picture/deblur/', num2str(j-1,'%d'), '.png'));
    gt = imread(target_file{1}{j});
    ssim_val = ssim(input, gt);
    psnr_val = psnr(input, gt);
    total_ssim = total_ssim + ssim_val;
    total_psnr = total_psnr + psnr_val;
end
qm_psnr = total_psnr / 1111;
qm_ssim = total_ssim / 1111;
fprintf('For %s dataset PSNR: %f SSIM: %f\n', datasets{1}, qm_psnr, qm_ssim);