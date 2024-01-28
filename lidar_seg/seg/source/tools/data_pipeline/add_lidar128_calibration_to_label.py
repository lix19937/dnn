# For rslidar128

import argparse
import os
import json

parser = argparse.ArgumentParser()
# e.g. --data_root xxx/labels/
parser.add_argument('--data_root', required=True, type=str)
# e.g. --json json from output of read_cam_intrinsic.py
parser.add_argument('--json', required=True, type=str)
# e.g. --idx set a number for identification
parser.add_argument('--idx', required=True, type=str)
args = parser.parse_args()

calibraion_v8_20210830 = {'rotation_matrix':
                          [['9.99917626e-01', '1.20339785e-02', '-4.46769595e-03'],
                           ['-1.20322574e-02', '9.99927521e-01', '4.11904912e-04'],
                              ['4.47232928e-03', '-3.58114514e-04', '9.99989927e-01']],
                          'rotation_angle': ['-3.85020307e-04', '-4.47013555e-03', '-1.20334486e-02'],
                          'translation_matrix': ['1.27828896e+00', '2.21237577e-02', '2.00611138e+00']}

calibraion_v2_20210714 = {'rotation_matrix':
                          [['9.99974847e-01', '6.98940502e-03', '1.22146867e-03'],
                           ['-6.96010888e-03', '9.99721825e-01', '-2.25360990e-02'],
                              ['-1.37864286e-03', '2.25270297e-02', '9.99745309e-01']],
                          'rotation_angle': ['2.25336608e-02', '1.30017672e-03', '-6.97540585e-03'],
                          'translation_matrix': ['1.26464546e+00', '-2.10034475e-02', '2.00707340e+00']}

calibraion_v14_20211022 = {
    'rotation_matrix':
    [["1", "0", "0"],
     ["0", "1", "0"],
     ["0", "0", "1"]],
    'rotation_angle': ["0", "0", "0"],
    'translation_matrix': ["1.25", "0.0", "2.06"]}

veh2cali = {
    '[1865.35340655084, 0, 1925.9439440757, 0, 1866.13488994884, 1095.36181663469, 0, 0, 1]': calibraion_v8_20210830,
    '[1865.3534065508, 0, 1925.9439440757, 0, 1866.1348899488, 1095.3618166347, 0, 0, 1]': calibraion_v8_20210830,
    '[1863.431125, 0, 1922.240153, 0, 1863.321656, 1079.249943, 0, 0, 1]': calibraion_v2_20210714,
    '[3127.914225854, 0, 1932.4000974847, 0, 3127.8976674782, 1105.8085011467, 0, 0, 1]': calibraion_v14_20211022,
}

with open(args.json, 'r') as f:
    time2intrin = json.load(f)

bad_labels = []

sub_dirs = [os.path.join(args.data_root, one, 'label')
            for one in os.listdir(args.data_root)]
for one_folder in sub_dirs:
    all_label_txt = [os.path.join(one_folder, one)
                     for one in os.listdir(one_folder)]
    for one_txt in all_label_txt:
        timestamp = one_txt.split('/')[-1].replace('.txt', '')
        if timestamp not in time2intrin:
            bad_labels.append(one_txt)
            continue
        print(one_txt)
        one_intrin = time2intrin[timestamp]
        cali = veh2cali[one_intrin]
        try:
            with open(one_txt, 'r') as f:
                label = json.load(f)
            label['calibration'] = cali
            with open(one_txt, 'w') as f:
                json.dump(label, f)
        except:
            bad_labels.append(one_txt)
with open('bad_labels'+args.idx+'.json', 'w') as f:
    json.dump(bad_labels, f)
