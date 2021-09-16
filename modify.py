import os

new_root = '/home/hh/Desktop/GoPro/'

for file_ in os.listdir('./datas/GoPro/before/'):
    read_path = os.path.join('./datas/GoPro/before/', file_)
    write_path = os.path.join('./datas/GoPro/', file_.replace('./datas/GoPro/before/', ''))
    f_r = open(read_path, 'r')
    f_w = open(write_path, 'w')


    for i in f_r.read().split('\n'):
        if i :
            new_path = os.path.join(new_root, i)
            f_w.write(new_path + '\n')

        

