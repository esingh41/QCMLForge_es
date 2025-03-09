export iter=4
python3 -u ./train_models.py \
    --train_apnet APNet3 \
    --am_model_path ./models/am_hf_ensemble/am_$iter.pt \
    --random_seed $iter \
    --lr 5e-5 \
    --ap_model_path ./models/ap3_ensemble/ap3_$iter.pt \
    --n_epochs 1 \
    --ds_max_size 100 \
