import os
def updatePath(newRoot):
    '''
        update the base read path with custom path
    '''
    for file_ in os.listdir('./datas/GoPro/base/'):
        read_path = os.path.join('./datas/GoPro/base/', file_)
        write_path = os.path.join('./datas/GoPro/', file_.replace('./datas/GoPro/base/', ''))
        f_r = open(read_path, 'r')
        f_w = open(write_path, 'w')

        for i in f_r.read().split('\n'):
            if i :
                new_path = os.path.join(newRoot, i)
                f_w.write(new_path + '\n')

if __name__ == "__main__":
    newRoot = '***'
    updatePath(newRoot)
