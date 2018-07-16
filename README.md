# Gender-Recognition-by-Voice
Using k-NN classifier to distinguish whether the voice sample is that of a male or female.
The elbow method was used to pick an optimim K-value(no. of neigbors)

Dataset available on Kaggle --> https://www.kaggle.com/primaryobjects/voicegender

Contents of repo:
* genderByVoice.py is the file which implements the classifier.
* voice.csv is the dataset.

Only the following 6 features out of 20 were considered:
* meanfreq
* sd
* centroid
* meanfun
* IQR
* median

The dataset contained 3168 records, 2376 of which were used as training set and 
the rest as test set.
