# SYDE675_Outlier_Detection
## GOAD with MCR2

## Requirements
* Python 3 +
* Pytorch 1.0 +
* Tensorflow 1.8.0 +
* Keras 2.2.0 +
* sklearn 0.19.1 +
## Changes
### We added the MCR2.py
### The changed files from GOAD repo are opt_tc.py and train_ad.py. The old codes were commented out.
### You can compare it with the opt_tc_old.py 


## Results
### The results of orginal GOAD with simple transformation is recorded in the results_CIFAR10_simple_trans.txt
### Our new results of widen factor 4,6,8, and 10 are recorded in results_CIFAR10_MCR2.txt


## Training

To replicate the results of the paper on CIFAR10 with widen factor 4:
```
python train_ad.py --m=0.1
```
New results will be saved in new text file called results_CIFAR10_MCR2_4.txt

### In order to change the widen factor, you can visit train_ad.py:
```
 parser.add_argument('--widen-factor', default=10, type=int)
```


