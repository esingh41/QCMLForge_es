import os

# t2 uses ap2 original ensemble for training, but on full dataset...

def create_sbatch_ap2(submit=False):
    for i in range(0, 5):
        fn = f"train_ap{i}.sbatch"
        #os.system(f"cp -r data_dir_ex data_{i}")
        python_call = """python3 -u ./train_models.py \\
    --train_am "AtomModel" \\
    --am_model_path ./models/am_neq/am_ensemble/am_$iter.pt \\
    --spec_type_am 7 \\
    --random_seed $iter \\
    --n_epochs 500 \\
    --lr 5e-5
    --data_dir ./data_$iter \\
        """
        with open(fn, 'w') as f:
            f.write(f"""#!/bin/bash
#SBATCH -J AM_non_eq_model_{i}
#SBATCH -o AM_non_eq_model_{i}_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH -pgpu-a100
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=esingh41@gatech.edu


cd /storage/home/hcoda1/3/awallace43/gits/qcmlforge/
source /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/3/awallace43/p-cs207-0/miniconda/envs/qcml

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