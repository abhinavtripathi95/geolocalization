# Scene Localization through Dual AlexNet Architecture

* This architecture is majorly based upon the scene localization 
network presented in [this paper](https://arxiv.org/abs/1809.05979).

## Code
* Run the jupyter notebook `train_dual.ipynb` for training the 
network.
* Dependencies: `torch torchvision os skimage numpy matplotlib`
* Dataset [source](https://uofi.app.box.com/s/4jfvpmxwiob0hcg25z4lgd5qgnk0q8nb): This example only requires you 
to download the uav and satellite images for `atlanta` city, and
the `data_labels` from the training set
* Runs on cuda enabled single GPU systems
* TODO: add support for CPU and multi-GPU systems

## Experiments and Results
* For easy formatting, I have pinned down theory, experimentation,
implementation details and results in python notebook `results.ipynb`.

<!-- 




This network uses two AlexNets without the last classification layer. 
The final output is the euclidean distance between the feature 
vectors of the two networks. Both the networks have different weights,
hence it is different from a Siamese network, where both the networks
have exactly the same weights.

This architecture is majorly based upon the scene localization 
network presented in [this paper](https://arxiv.org/abs/1809.05979).

### Loss Function
A contrastive loss function is used to separate matching images 
from non-matching images.
_ -->

