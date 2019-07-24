import sys
import os
from gen_fig.eval_quant import get_performance_wrap
from fileop import loadfile
import glob

RootPath = '/mnt/data/wding/tmp/bugs'
DetFile = 'detections_test1.pkl'
Beta = 2


def disp_func(run_path):
    dict_perf = get_performance_wrap(run_path, DetFile, Beta)

    print ''
    print '======================='
    print run_path
    config = loadfile(os.path.join(run_path, 'config.yaml'))
    for key in ['classifier_type', 'preproc_type']:
        print '{}: {}'.format(key, config[key])
        if config[key] == 'bowsvm':
            for key in config['bowsvm_config']:
                print '\t{}: {}'.format(key, config['bowsvm_config'][key])

    for key in ['logavg_miss_rate', 'precision_recall_auc']:
        print '{}: {}'.format(key, dict_perf[key])

if __name__ == '__main__':
    trial_name = sys.argv[1]
    run_path_list = glob.glob(os.path.join(RootPath, trial_name))

    for run_path in run_path_list:
        disp_func(run_path)
