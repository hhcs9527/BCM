import yaml
import os

def maker(epoch):
    channel_list = [16,32]
    epoch = epoch
    comb_list = [[True, True], [True, False]]

    with open('template.yaml') as f:
        experiment = yaml.load(f, Loader = yaml.FullLoader)
    
    for channel in channel_list:
        if os.path.exists('./' + str(channel) + '/') == False:
            os.system('mkdir ./' + str(channel) + '/') 

        for comb in comb_list:
            experiment['channel'] = channel
            experiment['train_epoch'] = epoch

            experiment['G'] = 'Patch_FPN_VAE'
            # if consider content
            if comb[1]:
                experiment['experiment_name'] = 'Patch_FPN_VAE_content_condiser_c' + str(channel)
                experiment['lambda_content'] = 1
            else:
                experiment['experiment_name'] = 'Patch_FPN_VAE_c' + str(channel)          
                experiment['lambda_content'] = 0  
        
            with open('./' + str(channel) + '/' + experiment['experiment_name'] + '.yaml', 'w') as file:
                documents = yaml.dump(experiment, file)

if  __name__ == "__main__":
    maker(500)