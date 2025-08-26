import os

# t2 uses ap2 original ensemble for training, but on full dataset...

def create_sbatch_am_non_eq(submit=False):
    for i in range(0, 5):
        fn = f"train_am_non_eq_cuda_{i}.sbatch"
        #os.system(f"cp -r data_dir data_{i}")
        python_call = """python3 -u ./train_models.py \\
    --train_am "AtomModel" \\
    --am_model_path ./models/am_neq/am_ensemble_cuda/am_$iter.pt \\
    --spec_type_am 7 \\
    --random_seed $iter \\
    --n_epochs 500 \\
    --lr 5e-5 \\
    --data_dir ./data_$iter \\
        """
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -J AM_non_eq_model_{i}
#SBATCH -o AM_non_eq_model_{i}_cuda_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH -pgpu-a100
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=esingh41@gatech.edu


cd /storage/home/hcoda1/1/esingh41/gits/QCMLForge_es
module load cuda/12.6.1
source /storage/home/hcoda1/1/esingh41/p-cs207-0/miniconda3/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/1/esingh41/p-cs207-0/miniconda3/envs/qcml

iter={i}
echo "
{python_call}
"
{python_call}
""")
    # --n_epochs 50 \\
    # --lr_decay 0.1 \\
        if submit:
            os.system(f'sbatch {fn}')

create_sbatch_am_non_eq()