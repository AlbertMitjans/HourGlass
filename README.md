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

**Testing log**
```
 * Recall(%): 6.250	 * Precision(%):  (6.818, 6.078, 5.191, 7.037)
```

The precision is computed for the (1, 2, 3, 4) corners detected with highest confidence (the gaussians with a highest value on its center).

## Run train

Trains the network from scratch or from a given ckpt.

```
$ python3 main.py
```

**Training log**
```
Epoch: [5][300/312]	Loss.avg: 0.3615	Recall(%): 21.622	Precision num. corners (%): (22.591, 18.563, 15.833, 16.809)
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
Watch the output of the network in realtime by connecting a depth camera.

Go to [this repository](https://github.com/AlbertMitjans/Xtion).

## Additional arguments
--train (default:True) : if True/False, training/testing is implemented.  
--val_data (default:True) : if True/False, all/validation data will be evaluated.  
--save_imgs (default:True) : if True output images will be saved in the \Output folder.  
--batch_size (default:1)  
--depth (default:True) : if True/False, depth/RGB images will be used.  
--ckpt(default:None)  
--num_epochs (default:200)  


