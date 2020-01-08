import os
import re
from shutil import copyfile

import numpy as np

job_name_pref = 'fna-retinanet-retrain-fna_model-refactorv1'
job_config = 2 # 0-customize 1-small  2-large
shell_name = 'k8s_job'

if job_config==0:
    num_pods = 4
    job_time = 2
    # titanv:10  idc_small:7  others:14 (days)
    cluster_num = 1
    # 1:idc 2:idc_small 3:share-titanv 4:share-2080
elif job_config==1: # small cluster config
    num_pods = 2
    job_time = 2
    cluster_num = 2
elif job_config==2: # large cluster config
    num_pods = 8
    job_time = 7
    cluster_num = 1
else:
    raise ValueError

def creat_submit_file(job_name_pref, num_pods, job_time, shell_name):
    config_name = 'k8s_config' + str(np.random.randint(99999)) +'.yaml'
    copyfile('./k8s_config_template.yaml', config_name)
    write_cache = ''

    with open(config_name, 'r') as f:
        for line in f.readlines():
            if '# JOB_NAME:' in line:
                continue
            elif 'JOB_NAME:' in line:
                write_cache += "  JOB_NAME: " + job_name_pref + '\n'
            elif 'GPU_PER_WORKER' in line:
                write_cache += '  GPU_PER_WORKER: ' + str(num_pods) + '\n' 
            elif 'WALL_TIME: ' in line:
                write_cache += '  WALL_TIME: ' + str(job_time) + '\n'
            elif 'RUN_SCRIPTS' in line:
                write_cache += '  RUN_SCRIPTS: ${WORKING_PATH}/scripts/' + str(shell_name) + '.sh\n'
            else:
                write_cache += line

    with open(config_name, 'w') as f:
        f.writelines(write_cache)
    
    return config_name


def modify_cluster_config(cluster_num):
    file_path = '/home/users/jiemin.fang/.hobot/gpucluster.yaml'
    write_cache = ''

    with open(file_path, 'r') as f:
        for line in f.readlines():
            if 'current-cluster' in line:
                write_cache += 'current-cluster: mycluster' + str(cluster_num)
            else:
                write_cache += line

    with open(file_path, 'w') as f:
        f.writelines(write_cache)


def submit(job_time, cluster_num, job_name_pref, num_pods, shell_name):
    job_time = job_time * 24 * 60
    modify_cluster_config(cluster_num)
    config_name = creat_submit_file(job_name_pref, num_pods, job_time, shell_name)
    os.system('traincli submit -f ' + config_name)
    os.system('rm ' + config_name)


if __name__ == "__main__":
    submit(job_time, cluster_num, job_name_pref, num_pods, shell_name)
