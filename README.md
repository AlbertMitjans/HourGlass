# Hourglass for corners detection of deformable fabric

This repo contains the code structure used for the detection of the corners of a wrinkled towel. 

## Installation

**Clone**
```
$ git clone https://github.com/AlbertMitjans/Hourglass.git
```
**Download pretrained weights**
```
???
```
**Download dataset**
```
???
```

## Run test

Evaluates the model on the dataset.

```
$ python3 main.py --train False
```

## Run train

Trains the network from scratch or from a given ckpt.

```
$ python3 main.py
```

**Tensorboard**

Track training progress in Tensorboard:
+ Initialize training
+ Run the command below
+ Go to [http://localhost:6006/](http://localhost:6006/)

```
$ tensorboard --logdir='logs' --port=6006
```


## Realtime test

Connecting a depth camera, the output of the network can be displayed in realtime.

```
$ python3 main.py --display True
```


## Additional arguments
--train (default:True) : if True/False, training/testing is implemented.  
--val_data (default:True) : if True/False, all/validation data will be evaluated.  
--save_imgs (default:True) : if True output images will be saved in the \Output folder.  
--plot_gradient (default:False) : if True the gradient plots will be saved in the \Output folder.  
--batch_size (default:1)  
--depth (default:True) : if True/False, depth/RGB images will be used.  
--ckpt(default:None)  
--num_epochs (default:200)  
--display (default:False) : activate realtime display of network's output  



