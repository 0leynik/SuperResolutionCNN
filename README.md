# SuperResolutionCNN
## Implementated on Keras

The original paper is [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)

## Result
- train data shape 1044750 x 1 x 32 x 32
- epoches 10
- batch_size 128

<p align="center">

<table style="width:100%">
<tr>
  <th>Original</th>
  <th>Downresized</th> 
  <th>Bicubic</th>
  <th>SRCNN</th>
</tr>
<tr>
  <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1.jpg"/></td>
  <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1_downresized.jpg"/></td> 
  <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1_bicubic.jpg"/></td>
  <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1_srcnn.jpg"/></td>
</tr>
</table>
</p>

