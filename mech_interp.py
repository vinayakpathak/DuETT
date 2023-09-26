import pytorch_lightning as pl

import duett

import physionet

from torchinfo import summary

import torch

seed=2020
pl.seed_everything(seed)
dm = physionet.PhysioNetDataModule(batch_size=512, num_workers=0,
                                   use_temp_cache=False)
dm.setup()
model = duett.Model.load_from_checkpoint('checkpoints/epoch=235-step=4012.ckpt', 
                                         d_static_num=dm.d_static_num(), 
                                         d_time_series_num=dm.d_time_series_num(), 
                                         d_target=dm.d_target(),
                                         pos_frac=dm.pos_frac(),
#                                         fusion_method='rep_token',
                                         seed=seed)

print(dm.d_static_num(), dm.d_time_series_num(), dm.d_target(), dm.pos_frac())
print(model.d_static_num, model.d_time_series_num, model.d_target, model.pos_frac)


trainer = pl.Trainer(gpus=1, max_epochs=50)

model.eval()

model.mode = 'interp'

trainer.test(model, dataloaders=dm)
