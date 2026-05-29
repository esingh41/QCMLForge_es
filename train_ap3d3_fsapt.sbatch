#!/bin/bash
#SBATCH -JAP3D3-SD-4
#SBATCH -oAP3D3-SD-4_training.out
#SBATCH -Agts-cs207-chemx
#SBATCH --open-mode=append
#SBATCH -N1 --ntasks=1 --cpus-per-task=8 -G1
#SBATCH -pgpu-v100
#SBATCH --mem-per-cpu=12G
#SBATCH -t72:00:00
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=esingh41@gatech.edu

cd /storage/home/hcoda1/1/esingh41/r-cs207-0/gits/qcmlforge
source /storage/home/hcoda1/1/esingh41/r-cs207-0/miniconda3/etc/profile.d/conda.sh
conda activate /storage/home/hcoda1/1/esingh41/r-cs207-0/miniconda3/envs/qcml

set -euo pipefail

MODEL_DIR=./models/ap3_fsaptdft/
SCRATCH_ROOT="${TMPDIR:-/tmp}/ap3_fsaptdft"
LOCAL_DATA_DIR="${SCRATCH_ROOT}/data_dir"
mkdir -p "${MODEL_DIR}" "${SCRATCH_ROOT}"

sync_dataset_to_tmp() {
    local src="$1"
    local dst="$2"
    shift 2
    if [ ! -d "$src/processed" ]; then
        echo "Expected processed dataset under $src/processed" >&2
        exit 1
    fi
    rm -rf "$dst"
    mkdir -p "$dst/raw" "$dst/processed"
    if [ -d "$src/raw" ]; then
        shopt -s nullglob
        for raw_path in "$src"/raw/*; do
            touch "$dst/raw/$(basename "$raw_path")"
        done
        shopt -u nullglob
    fi
    for raw_name in "$@"; do
        touch "$dst/raw/$raw_name"
    done
    cp -a "$src/processed/." "$dst/processed/"
}


export iter=1
python3 \
    -u \
    ./train_models.py \
    --train_apnet \
    APNet3-fused-d3 \
    --am_model_path \
    ./models/ap3_ensemble/$iter/am_3.pt\
    # ./models/ap3_saptpbe0/4/am_ap2_4.pt \
    --atom_type_param_model_path \
    ./models/ap3_ensemble/$iter/am_h+1_3.pt
    # ./models/ap3_saptpbe0/4/atp_hfvr_4.pt \
    --atom_type_param_model_path2 \
    ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \
    # ./models/ap3_saptpbe0/4/atp_elst_4.pt \
    --ap_pretrained_model_path
    ./models/ap3_ensemble/$iter/ap3_.pt \
    --random_seed \
    4 \
    --ap_model_path \
    ./models/ap3_ensemble/$iter/ap3d3_fsaptdft.pt \
    --n_epochs \
    50 \
    --n_rbf \
    8 \
    --n_neuron \
    128 \
    --n_embed \
    8 \
    --data_dir \
    ./data_dir_saptpbe0 \
    --spec_type_ap \
    10 \
    --ds_type fsapt_energies \
    --lr \
    5e-5 \
    --ds_class_type \
    lmdb \
    --unfreeze_dimer_prop_model \
    --unfreeze_atom_model \
    --build_dataset_only
sync_dataset_to_tmp ./data_dir_saptpbe0 "${LOCAL_DATA_DIR}" 189K_saptpbe0-d4_totals_train.pkl 189K_saptpbe0-d4_totals_test.pkl

python3 \
    -u \
    ./train_models.py \
    --train_apnet \
    APNet3-fused-d3 \
    --am_model_path \
    ./models/ap3_ensemble/$iter/am_3.pt\
    # ./models/ap3_saptpbe0/4/am_ap2_4.pt \
    --atom_type_param_model_path \
    ./models/ap3_ensemble/$iter/am_h+1_3.pt
    # ./models/ap3_saptpbe0/4/atp_hfvr_4.pt \
    --atom_type_param_model_path2 \
    ./models/ap3_ensemble/$iter/am_elst_h+1_3.pt \
    # ./models/ap3_saptpbe0/4/atp_elst_4.pt \
    --ap_pretrained_model_path
    ./models/ap3_ensemble/$iter/ap3_.pt \
    --random_seed \
    4 \
    --ap_model_path \
    ./models/ap3_ensemble/$iter/ap3d3_fsaptdft.pt \
    --n_epochs \
    50 \
    --n_rbf \
    8 \
    --n_neuron \
    128 \
    --n_embed \
    8 \
    --data_dir \
    ${LOCAL_DATA_DIR} \
    --spec_type_ap \
    10 \
    --ds_type fsapt_energies \
    --lr \
    5e-5 \
    --ds_class_type \
    lmdb \
    --unfreeze_dimer_prop_model \
    --unfreeze_atom_model \


