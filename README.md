# SuperResolutionCNN
## Implementated on Keras

The original paper is [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)

## Result
- train data: 1044750 x 1 x 32 x 32 imgs
- epoches: 10
- batch_size: 128

<table style="width:100%" align="center">
  <tr>
    <th>Original</th>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1.jpg" width="400"/></td>
  </tr>
  <tr>
    <th>Downresized</th>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1_downresized.jpg" width="300"/></td> 
  </tr>
  <tr>
    <th>Bicubic</th>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1_bicubic.jpg" width="200"/></td>
  </tr>
  <tr>
    <th>SRCNN</th>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1_srcnn.jpg" width="150"/></td>
  </tr>
</table>

