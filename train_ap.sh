iter=4
python3 -u ./train_models.py \
    --train_am AtomHirshfeldModel \
    --am_model_path ./models/am_hf_ensemble/am_$iter.pt \
    --spec_type_am 1 \
    --n_epochs_atom 2 \
    --data_dir ./data_$iter \
    --data_dir_atom ./data_$iter \
    --random_seed $iter \
