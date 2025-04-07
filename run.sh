iter=0
python -u ./train_models.py \
    --am_model_path ./models/am_ensemble/am_$iter.pt \
    --data_dir ./data_$iter \
    --data_dir_atom ./data_$iter \
    --random_seed $iter \
    --train_ap APNet2 \
    --ap_model_path ./models/ap2_ensemble/ap2_t1_$iter.pt \
    --n_epochs 50 \
    --spec_type_ap 2 \
    --lr_decay 0.1 \
    --lr 0.002
