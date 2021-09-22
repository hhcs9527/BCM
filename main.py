import yaml
import argparse
from datetime import datetime
from Trainer import Trainer

class config():
    def __init__(self, experiment):
        # training setting
        self.isTrain = experiment['isTrain']
        self.epochs = experiment['train_epoch']
        self.start_epoch = experiment['start_train_epoch']
        self.batchsize = experiment['batch_size']
        self.imagesize = experiment['image_size']
        self.learning_rate = experiment['learning_rate']
        self.gpu = experiment['gpu']
        
        # deblur net setting
        self.G = experiment['G']
        self.channel = experiment['channel']

        # reblur setting
        self.Recurrent_times = experiment['Recurrent_times']
        self.per_pix_kernel = experiment['per_pix_kernel']


if __name__ == '__main__':
    # for usging shell to run different config    
    #default_path = './experiment_config/32/BRRM_DMPHN_1_2_4_c32.yaml'
    default_path = './experiment_config/32/Sharp.yaml'
    default_path = './experiment_config/32/RDMPHN.yaml'
    #default_path = './experiment_config/32/BRRMv5_DMPHN_1_2_4_8_c32.yaml'
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("-conf","--config",type=str, default = default_path)
    config_args = config_parser.parse_args()

    with open(config_args.config) as f:
        experiment = yaml.load(f, Loader = yaml.FullLoader)

    args = config(experiment)              
    print(experiment)
    
    #Hyper Parameters
    METHOD = experiment['experiment_name']
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    GPU = args.gpu
    BATCH_SIZE = args.batchsize
    IMAGE_SIZE = args.imagesize

    #args.isTrain = False
    MP = Trainer(args, experiment, METHOD, LEARNING_RATE, EPOCHS, GPU, BATCH_SIZE, IMAGE_SIZE)
    print('Running ... {} epochs !!!! , time : {}'.format(args.epochs, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    if args.isTrain:
        #model = MP.MPGAN.netG
        #print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000)
        #MP.train()
        MP.generate_eval_pic()
        #for e in [i * 100 for i in range(1, 31)]:
        #    MP.visualize_motion_feature_map(e)
        #MP.visualize_motion_feature_map()
        #MP.visualize_level_output()
        #MP.eval_checkpoints()
    else:
        print('saving')
        MP.lib_calculate_metric()#generate_eval_pic()