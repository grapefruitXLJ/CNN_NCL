# Method of combining one-dimensional convolution neural network and negative correlation learning for analysis of near-infrared spectra
## Dataset
* [NIR of Corn Sample for Standardization Benchmarking](http://www.eigenvector.com/data/Corn/index.html) <br>
> m5 data was chosen to build the model. In addition, the outliers (75 and 77) are removed from this dataset. Ramdonly choose 62 sampels for cailbration,8 for vaildation and 8 for prediction.Since there are only 78 samples in the dataset, we preform SMOTE algorithm on it.
* [Near Infrared Spectra of Diesel Fuels](http://www.eigenvector.com/data/SWRI/index.html)<br>
 > There are three formats of these data. Standard Matlab Variable Format was used.
## SMOTE
> Synthetic Minority Over-sampling Technique (SMOTE) is used to expand corn dataset.N/100 is the sampling ratio and k is the number of neighbors.it is not reliable to build a model using data created by SMOTE, so it is necessary to use real samples for prediction.Before SMOTE, extend Y(Independent variable) to the back of X(dependent variable).

## Sub-network
>This one-dimensional convolution neural network is suitable for both spectral data mentioned above.  

![](https://github.com/grapefruitXLJ/CNN_NCL/blob/master/cnn%20structure.jpg)

## Training skill
The model cannot be trained at once. It is necessary to adjust the learning rate and the number of epochs according to the changes of rmsec and rmsecv. The picture below shows that the model has converged and it's time to stop training.

![](https://github.com/grapefruitXLJ/CNN_NCL/blob/master/Changes%20of%20RMSEC%20and%20RMSECV.png)


