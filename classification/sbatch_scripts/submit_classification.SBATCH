#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=7:00:00
#SBATCH --mem=100GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=cds

module purge

singularity exec --nv \
        --overlay $SCRATCH/backup_playground_overlay.ext3:ro \
        --overlay $SCRATCH/LIDC-HDF5-256.sqf:ro \
        /scratch/work/public/singularity/cuda10.1-cudnn7-devel-ubuntu18.04-20201207.sif \
        /bin/bash -c "source /ext3/env.sh; cd /scratch/es5223/medsynth/classification/Tuberculosis_ChestXray_Classifier/Notebooks/; \
	python -u $1 $2 "

