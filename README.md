# Real-time display (Python2.7)
This repo contains the code structure to display the output of a neural network when using images from a connected camera as input.

## How to use it  

**Connect camera**  
Connect an Asus XtionPRO LIVE camera and launch it as follows:
```
roslaunch openni2_launch openni2.launch depth_registration:=true
```

**Run**
```
realtime_display.py --ckpt1 CKPT1 --ckpt2 CKPT2 --compare TRUE/FALSE
```

**Arguments**  
--ckpt1 (default="best_checkpoints/ckpt_1_1.pth") : path to checkpoint/model file  
--ckpt2 (default="best_checkpoints/ckpt_11.pth") : path to checkpoint/model file (only used when compare=True)  
--compare (default=False) : if True the outputs of ckpt1 and ckpt2 will be displayed  
