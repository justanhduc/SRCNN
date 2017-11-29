# SRCNN
An implementation of [SRCNN](https://arxiv.org/abs/1501.00092)

## Requirements
[Theano v0.9](http://deeplearning.net/software/theano/)

[NeuralNet](https://github.com/justanhduc/neuralnet)

## Result
The model is trained using the scheme described in the paper with some small simplifications. 

![Origin](https://github.com/justanhduc/SRCNN/blob/master/figures/ILSVRC2012_test_00007795.JPEG)
Original Image

![Bicubic](https://github.com/justanhduc/SRCNN/blob/master/figures/ILSVRC2012_test_00007795_bicubic.JPEG)
Bicubic interpolation

![SRCNN](https://github.com/justanhduc/SRCNN/blob/master/figures/ILSVRC2012_test_00007795_srcnn.JPEG)
SRCNN interpolation

## Usages
To train SRCNN using the current training scheme

```
python train_srcnn.py
```

To test SRCNN with an image
```
python upsample.py image_destination
```


## Credits
[C. Dong, C. C. Loy, K. He, and X. Tang, "Image super-resolution using deep convolutional networks," IEEE transactions on pattern analysis and machine intelligence, vol. 38, pp. 295-307, 2016.](https://arxiv.org/abs/1501.00092)
