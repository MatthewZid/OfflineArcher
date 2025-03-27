#!/bin/bash -l
# SLURM SUBMIT SCRIPT

#SBATCH --nodelist=node-gpu01
#SBATCH --gres=gpu:1    # Request N GPUs per machine

# . init.sh

lr=1e-4
# batch_size=32
batch_size=4
accumulate_grad_batches=4 #8

python main.py fit \
--data=Socratic \
--data.batch_size=$batch_size \
--data.n_traj_eval=64 \
--model=BehaviouralCloning \
--model.lr=$lr \
--trainer.fast_dev_run=False \
--trainer.max_epoch=13 \
--trainer.accumulate_grad_batches=$accumulate_grad_batches \
--trainer.logger=TensorBoardLogger \
--trainer.logger.save_dir='lightning_logs' \
--trainer.limit_val_batches=0.0 \
--trainer.devices=[1]