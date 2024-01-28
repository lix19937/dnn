import os
import sys
this_dir = os.path.dirname(__file__)
last_dir = os.path.dirname(this_dir)

# Add lib to PYTHONPATH
if last_dir not in sys.path:
    sys.path.insert(0, last_dir)
from source.lib.datasets.evaluate.lidar_seg_eval import plot_loss

if __name__ == '__main__':
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    save_path = plot_loss(root_dir, root_dir)