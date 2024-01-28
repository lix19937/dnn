# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

from superresolutionnet import SuperResolutionNet
from qat_base import NetOpts, pipeline, fix_seed
from loguru import logger

if __name__ == "__main__":
    fix_seed()

    opts = NetOpts() 
    opts.model_name = 'SuperResolutionNet'
    opts.input_dims = (1, 1, 244, 244)
    opts.nn_module = SuperResolutionNet(upscale_factor=3)

    pipeline(opts)
    logger.info("all done")
