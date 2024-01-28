# For luminar.

import json
import os


root_dir = '/home/igs/Downloads/labels'
calibraion = {'rotation_matrix': [['9.99924839e-01', '-1.13726659e-02', '-4.58624261e-03'],
                                  ['1.14255212e-02', '9.99866664e-01',
                                      '1.16680861e-02'],
                                  ['4.45293356e-03', '-1.17196087e-02', '9.99921381e-01']],
              'rotation_angle': ['-1.16944071e-02', '-4.51980438e-03', '1.13996388e-02'],
              'translation_matrix': ['1.66035831e+00', '3.09985187e-02', '1.61256278e+00']}

sub_dirs = os.listdir(root_dir)
for one_folder in sub_dirs:
    one_folder = os.path.join(root_dir, one_folder)
    jsons = [os.path.join(one_folder, one_json)
             for one_json in os.listdir(one_folder)]
    for one_label in jsons:
        with open(one_label, 'r') as f:
            label = json.load(f)
        if 'calibration' in label:
            continue
        label['calibration'] = calibraion
        with open(one_label, 'w') as f:
            json.dump(label, f)
        print(one_label)
