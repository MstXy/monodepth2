# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    if opts.ddp:
        import os
        import torch.multiprocessing as mp
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        world_size = 4
        raise NotImplementedError
        # mp.spawn(use_ddp,
        #          args=(world_size, ),
        #          nprocs=world_size,
        #          join=True)
    else:
        trainer = Trainer(opts)
        trainer.train()

