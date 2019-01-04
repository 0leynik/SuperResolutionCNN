# SuperResolutionCNN (impl on Keras)

The original paper is [Learning a Deep Convolutional Network for Image Super-Resolution](https://arxiv.org/abs/1501.00092)

## Dataset
[DIV2K High Resolution Images](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- [Train Data (HR images)](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip)
- [Validation Data (HR images)](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip)

## Result on 81 000 iteration
### training settings
- input train data: 1 044 750 x 32 x 32 x 1 imgs
- output train data: 1 044 750 x 20 x 20 x 1 imgs
- epoches: 10
- batch_size: 128
- optimizer: Adam, lr=0.0003
- loss: MSE

<table style="width:100%" align="center">
  <tr>
    <th>Original</th>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1.jpg" height="250"/>
    <div align="center">514 × 343</div>
    </td>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/butterfly.png" height="250"/>
    <div align="center">256 × 256</div>
    </td>
  </tr>
  <tr>
    <th>Downresized<br>factor 2x</th>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1_downresized.jpg" height="250"/>
    <div align="center">257 × 171</div>
    </td>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/butterfly_downresized.png" height="250"/>
    <div align="center">128 × 128</div>
    </td>
  </tr>
  <tr>
    <th>Bicubic interpolation</th>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1_bicubic.jpg" height="250"/>
    <div align="center">514 × 343</div>
    <div align="center">PSNR = 34.36</div>
    </td>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/butterfly_bicubic.png" height="250"/>
    <div align="center">256 × 256</div>
    <div align="center">PSNR = 24.76</div>
    </td>
  </tr>
  <tr>
    <th>SRCNN</th>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/1_srcnn.jpg" height="250"/>
    <div align="center">514 × 343</div>
    <div align="center">PSNR = 37.12</div>
    </td>
    <td><img src="https://github.com/0leynik/SuperResolutionCNN/blob/master/predict_2x/butterfly_srcnn.png" height="250"/>
    <div align="center">256 × 256</div>
    <div align="center">PSNR = 30.45</div>
    </td>
  </tr>
</table>
