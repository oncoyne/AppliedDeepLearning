#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-00:30
#SBATCH --account comsm0045
#SBATCH --mem 16GB
#SBATCH --gres gpu:1

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

# for i in {0..4}
# do
#     python visualisation.py --preds "preds/Shallow_CNN_Salicon_run_0/validation_round_$i/preds.pkl" --gts "val.pkl"
# done
python visualisation.py --preds "preds/Shallow_CNN_Salicon_run_0/validation_round_500/preds.pkl" --gts "val.pkl"  