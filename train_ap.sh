# export iter=0
# python3 -u ./train_models.py \
#     --train_apnet APNet3 \
#     --am_model_path ./models/am_hf_ensemble/am_$iter.pt \
#     --random_seed $iter \
#     --lr 5e-5 \
#     --ap_model_path ./models/ap3_ensemble/ap3_$iter.pt \
#     --n_epochs 1 \
#     --ds_max_size 100 \

export iter=0
python3 -u ./train_models.py \
    --train_apnet APNet2 \
    --am_model_path ./models/am_ensemble/am_$iter.pt \
    --random_seed $iter \
    --lr 5e-5 \
    --ap_model_path ./models/dapnet2/ap2_$iter.pt \
    --n_epochs 1 \
    --r_cut_im 16.0 \
    --data_dir ./data_dir_dapnet
    # --ds_max_size 100 \

m1="B3LYP-D3/aug-cc-pVDZ/CP"
m1="B3LYP-D3/aug-cc-pVTZ/CP"
m2="CCSD(T)/CBS/CP"
m1_str="B3LYP-D3_aug-cc-pVDZ_CP"
m1_str="B3LYP-D3_aug-cc-pVTZ_CP"
m2_str="CCSD_LP_T_RP_CBS_CP"
output_name="${m1_str}_to_${m2_str}_${iter}.pt"
rm -r ./data_dir/processed_delta/

# --ds_max_size 100 \
python3 -u ./train_models.py \
    --train_apnet dAPNet2 \
    --am_model_path ./models/am_ensemble/am_$iter.pt \
    --random_seed $iter \
    --lr 5e-4 \
    --ap_model_path ./models/dapnet2/$output_name \
    --n_epochs 5 \
    --spec_type_ap 2 \
    --m1 $m1 \
    --m2 $m2 \
    --data_dir ./data_dir_dapnet
