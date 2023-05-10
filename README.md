# Decoders based on neural networks used to decode reaching and reach-to-grasping from the posterior parietal cortex using deep neural networks

The codes of this repository are associated to the journal paper _Motor decoding from the posterior parietal cortex using deep neural networks_ by Davide Borra, Matteo Filippini, Mauro Ursino, Patrizia Fattori, and Elisa Magosso. Journal of Neural Engineering (2023).
In that study, we compared different deep neural networks (fully-connected neural networks, convolutional neural networks, and recurrent neural networks) when decoding motor states from the neural activity of the posterior parietal cortex of macaques. 

Three different motor decoding tasks were addressed: reaching and reach-to-grasping decoding (this one with two different illumination conditions), involving the classification of different reaching end-points or different grip shapes. 
These motor tasks are referred as: _reaching_, _reach-to-grasping light_ (i.e., performed with good illumination conditions) and _reach-to-grasping dark_ (i.e., performed in darkness).
Four different monkeys performed the tasks (m1-m4).

The used neural networks are defined in the 'models.py' script. 
The optimal neural network designs were searched using Bayesian optimization, separately for the different neural networks and decoding problems. 
The optimal designs are contained in the pickle file 'optimal_hparams.pkl'.  

The decoding performance was investigated under different training conditions, also by artificially reducing the datasets, reflecting different recording scenarios. Furthermore, task-to-task transfer learning was also investigated.

The 'main.py' script is a sample script showing how to use the trained networks with the optimal designs (as resulted from Bayesian optimization).

### Prerequisites
* PyTorch

Please cite our manuscript if you use our models or results for your research.

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
