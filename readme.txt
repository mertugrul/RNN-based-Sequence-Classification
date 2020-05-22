## Train the model. It will save the .pkl file under Results/
## --cw: class weight [no | yes]. It works for survived patients.
## --dropout: dropout value (works if --n_layers > 1)
## --hidden_dim: hidden dimension for LSTM.
## --n_layers: number of LSTM layers to be stacked.
## --lr: learning rate
python3 classify.py --dropout 0.2 --hidden_dim 32 --n_layers 1 --lr 0.00005 --cw no

## Evaluate the model (paramaters are the same as classify.py). It will save the .csv file under Results/
python3 evaluate.py --dropout 0.2 --hidden_dim 32 --n_layers 1 --lr 0.00005 --cw no