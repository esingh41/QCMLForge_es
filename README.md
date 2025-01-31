# QCMLForge

Provides QCArchive data for creating QC ML models. The AP-Net2 has been re-implemented in PyTorch with newer versions to come.

# PyTorch AP-Net2 
Code re-implemented from TensorFlow version located [here](https://github.com/zachglick/apnet)

## Training
To train the model, run the following command:
```bash
python3 ./train_models.py \
    --train_ap2 \
    --ap_model_path ./models/example/ap2_example.pt \
    --n_epochs 5 
```

# PyTorch AtomicModule 
Code re-implemented from TensorFlow version located [here](https://github.com/zachglick/apnet)

## Training
To train the model, run the following command:
```bash
python3 ./train_models.py \
    --train_am \
    --am_model_path ./models/example/am_example.pt \
    --n_epochs 5 
```
