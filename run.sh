iter=0
m1="B3LYP-D3/aug-cc-pVDZ/CP"
m2="CCSD(T)/CBS/CP"
m1_str="B3LYP-D3_aug-cc-pVDZ_CP"
m2_str="CCSD_LP_T_RP_CBS_CP"
output_name="${m1_str}_to_${m2_str}_${iter}.pt"

python3 -u ./train_models.py \
    --train_apnet dAPNet2 \
    --am_model_path ./models/am_ensemble/am_$iter.pt \
    --random_seed $iter \
    --lr 5e-4 \
    --ap_model_path ./models/dapnet2/$output_name \
    --n_epochs 50 \
    --spec_type_ap 2 \
    --m1 $m1 \
    --m2 $m2 \
