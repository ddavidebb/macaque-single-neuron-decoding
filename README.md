# Deep neural networks for decoding reaching and reach-to-grasping from the posterior parietal cortex

The codes of this repository are associated to the journal papers:
* _Motor decoding from the posterior parietal cortex using deep neural networks_ by Davide Borra, Matteo Filippini, Mauro Ursino, Patrizia Fattori, and Elisa Magosso. Journal of Neural Engineering (2023).
* _Convolutional neural networks reveal properties of reach-to-grasp encoding in posterior parietal cortex_ by Davide Borra, Matteo Filippini, Mauro Ursino, Patrizia Fattori, and Elisa Magosso. Computers in Biology and Medicine (2024), currently under review (citing information available soon).

## Motor decoding from the posterior parietal cortex using deep neural networks
In this study, we compared different deep neural networks (fully-connected neural networks, convolutional neural networks, and recurrent neural networks) when decoding motor states from the neural activity of the posterior parietal cortex of macaques. 

Three different motor decoding tasks were addressed: reaching and reach-to-grasping decoding (this one with two different illumination conditions), involving the classification of different reaching end-points or different grip shapes. 
These motor tasks are referred as: _reaching_, _reach-to-grasping light_ (i.e., performed with good illumination conditions) and _reach-to-grasping dark_ (i.e., performed in darkness).
Four different monkeys performed the tasks (m1-m4).

### Usage
The used neural networks are defined in the 'models.py' script. 
The optimal neural network designs were searched using Bayesian optimization, separately for the different neural networks and decoding problems. 
The optimal designs are contained in the pickle file 'optimal_hparams.pkl'.  

The decoding performance was investigated under different training conditions, also by artificially reducing the datasets, reflecting different recording scenarios. Furthermore, task-to-task transfer learning was also investigated.

The 'main.py' script is a sample script showing how to use the trained networks with the optimal designs (as resulted from Bayesian optimization).

## Convolutional neural networks reveal properties of reach-to-grasp encoding in posterior parietal cortex
In this study, we used a convolutional neural network for defining an explainable AI framework for analyzing the most relevant cells and time samples involved during reach-to-grasping, from posterior parietal cortex recordings of macaques.

The convolutional neural network design is inspired from an architecture originally proposed for EEG signals (EEGNet, Lawhern et al., 2018, see 'Other references' Section) and was adapted here for decoding single neurons. The network - termed 'FiringRateNet' (FRNet) - was used to decode reach-to-grasping from the posterior parietal cortex (V6A area) of macaques. Furthermore, FRNet decision was explained by using layerwise relevance propagation.

### Usage
The 'model.py' script contains the code needed to design the network.
To use this network, data should be arranged as 3-D input maps of shape (1, n_cells, n_time), where n_cells and n_time denote the number of recorded cells and time samples, respectively. Assuming that data is prepared in this way, with n_cells=93, n_time=60, and n_classes=5 (denoting with n_classes the number of conditions to classify):

```
from model import FRNet
from torchsummary import summary

model = FRNet(n_chans=93, n_classes=5, n_time=60)
summary(model, input_size=(1, 93, 60))
```

## Prerequisites
* PyTorch
* torchsummary

Please cite our manuscripts if you use our models or results for your research.

## Citing
```bibtex
@article{10.1088/1741-2552/acd1b6,
	author={Borra, Davide and Filippini, Matteo and Ursino, Mauro and Fattori, Patrizia and Magosso, Elisa},
	title={Motor decoding from the posterior parietal cortex using deep neural networks},
	journal={Journal of Neural Engineering},
	url={http://iopscience.iop.org/article/10.1088/1741-2552/acd1b6},
	year={2023},
}
```
Available soon for _Convolutional neural networks reveal properties of reach-to-grasp encoding in posterior parietal cortex_ by Davide Borra, Matteo Filippini, Mauro Ursino, Patrizia Fattori, and Elisa Magosso. Computers in Biology and Medicine (2024).

## Other references
```bibtex
@article{lawhern2018,
doi = {10.1088/1741-2552/aace8c},
url = {https://dx.doi.org/10.1088/1741-2552/aace8c},
year = {2018},
month = {jul},
publisher = {IOP Publishing},
volume = {15},
number = {5},
pages = {056013},
author = {Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
title = {EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces},
journal = {Journal of Neural Engineering},
}
```
