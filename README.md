# Real-time display (Python2.7)
This repo contains the code structure to display the output of a neural network when using images from a connected camera as input.

## How to use it  

**Clone**  
```
$ git clone https://github.com/AlbertMitjans/real-time.git
```

**Download pre-trained weights**
```
$ cd checkpoints
$ bash get_weights.sh
```

**Connect camera**  
Connect an Asus XtionPRO LIVE camera and launch it as follows:
```
$ roslaunch openni2_launch openni2.launch depth_registration:=true
```

**Run**
```
$ python2 realtime_display.py --ckpt1 CKPT1 --ckpt2 CKPT2 --compare TRUE/FALSE
```
Press "s" to save an image.
Press "q" to quit.

**Arguments**  
--ckpt1 (default="checkpoints/ckpt.pth") : path to checkpoint/model file  
--ckpt2 (default="checkpoints/ckpt_2.pth") : path to checkpoint/model file (only used when compare=True)  
--compare (default=False) : if True the outputs of ckpt1 and ckpt2 will be displayed  
