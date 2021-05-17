# SEEG-Adversarial Neural Network model(SEEG-ANN)

## Environment

The model built by Pytoch framework, Python3.6. You May need to install these packages as follows: 

dtw               1.4.0

mne               0.23.dev0

numpy             1.19.5

opencv-python     4.5.1.48

pandas            1.2.0

pyEDFlib          0.1.20

pyparsing         2.4.7

python-dateutil   2.8.1

scikit-learn      0.24.0

scipy             1.6.0

torch             1.7.1

torchvision       0.8.2

tqdm              4.56.0

## How to run?

You can use shell to run code, and you need go to "mymodel" dir.

Training  EDANN-Transformer model by:

```
python ./run.py -m train -lac Transformer -p patient_name -gpu 0 -ep 30 -bs 64
```

Test EDANN-Transformer model by:

```
python ./run.py -m test -lac Transformer -p patient_name -gpu 0 
```

Training  EDANN-LSTM model by:

```
python ./run.py -m train -lac LSTM -p patient_name -gpu 0 -ep 30 -bs 64
```

Test EDANN-LSTM model by:

```
python ./run.py -m test -lac LSTM -p patient_name -gpu 0 
```



