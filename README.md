# deep_retina_pytorch
Reimplementation of deep_retina in Pytorch

[Paper](https://arxiv.org/abs/1702.01825) and original [code](https://github.com/baccuslab/deep-retina) in Keras

## Requirements
* Pytorch
* tensorboardx
* numpy
* matplotlib and seaborn

## Results
For this results, I trained the model on the data from [paper](https://www.sciencedirect.com/science/article/pii/S0896627316302501) "Analysis of Neuronal Spike Trains, Deconstructed, by Aljadeff et. al. 2016" with some arbitrary hyperparameters. For better results, more hyperparameter tuning is needed.

#### Spike count prediction sample
![Spike count prediction sample](https://github.com/bmeatayi/deep_retina_pytorch/blob/dev/images/spike_cnt_pred.png)


#### Filters sample
![Filters sample](https://github.com/bmeatayi/deep_retina_pytorch/blob/dev/images/filter_sample.png)


## TO DO
- [x] CNN implementation
- [x] Parametric SoftPlus
- [x] L1 and L2 regularization
- [x] Visualization
- [ ] Recurrent NN
- [ ] Hyperparameter tuning
