# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

from resnet50 import ResNet50
from qat_base import NetOpts, pipeline, fix_seed
from loguru import logger

if __name__ == "__main__":
    fix_seed()

    opts = NetOpts() 
    opts.model_name = 'ResNet50'
    opts.input_dims = (1, 3, 224, 224)
    opts.nn_module = ResNet50(num_classes=1000)

    pipeline(opts)
    logger.info("all done")
