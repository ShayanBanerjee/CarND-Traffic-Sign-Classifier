# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./images/sign1.jpg "sign1"
[image9]: ./images/sign2.jpg "sign2"
[image10]: ./images/sign3.jpg "sign3"
[image11]: ./images/sign4.jpg "sign4"
[image12]: ./images/sign5.jpg "sign5"
[image13]: ./images/sign6.jpg "sign6"
[image14]: ./images/sign7.jpg "sign7"
[image15]: ./images/sign8.jpg "sign8"
[image16]: ./examples/exploratory.png "exploratory"
[image17]: ./examples/prediction.png "predictions"
[image18]: ./examples/sample.png "Total samples exploration"


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

figure shows the distribution of the traffic signal labels from the training dataset in the form of a histogram. The occurrence of labels between 2 and 15 is higher than the rest of the set. 

![alt text][image18]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the three color channels do not add to significant information as input to the CNN

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because this brings the mean of the data close to 0. Literature says that normalization makes training easier and speeds it up.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, output 28x38x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, output 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6    				|
| Fully connected		| 400 (input) to 120 (output)					|
| RELU					|												|
| Dropout        		| Randomly dropout some units during training 	|
| Fully connected		| 120 (input) to 84 (output)					|
| RELU					|												|
| Dropout        		| Randomly dropout some units during training 	|
| Final Fully connected	| 84 (input) to 43 (output)						|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model was trained using the "AdamOptimizer"  with a batch size = 128 and epochs = 30 and a learning rate of 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.9
* validation set accuracy of 92.3
* test set accuracy of 91.4

If an iterative approach was chosen:
* I started exactly with the LeNet architecture provided in the class material.
*This did not work as well
*I added the two dropouts after the fully connected layers to avoid overfitting.
*I had to increse the number of epochs to 30 which seemed to give the best results.
* I played around with the batch size a little and settled on 128 as it gave the best results


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image8]![alt text][image9] ![alt text][image10]
![alt text][image11] ![alt text][image12] ![alt text][image13] 
![alt text][image14]![alt text][image15] 

The first image might be difficult to classify because the human is clear but not his purpose 
The second image might be difficult to classify because the lines by car is intersecting
The third image might be difficult only because of image resolution present here 
The 4th image might be difficult to classify as the line direction could be misunderstood from distance
The 5th  image might be difficult to classify as it has low chances of detection
The 6th image might be difficult to classify as the exclaimation structure can look like straight ahead sign
The 7th  image might be difficult to classify as the bent could have been more sharp
The 8th image might be difficult to classify as it may not get detected at all

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image17]


The model was able to correctly classify all the selected test images

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model picks the correct label with very high probability (more than 5 decimal places) for all the images. The model predicts a wrong label for the third image (30 km/hr) with a probability of 0.00001. 

The table below presents the top 5 softmax probabilities for each image along with the corresponding labels.


Predictions:
 [[ 1.       0.       0.       0.       0.     ]
 [ 0.80646  0.19353  0.       0.       0.     ]
 [ 0.905    0.09483  0.00016  0.       0.     ]
 [ 0.99987  0.00013  0.       0.       0.     ]
 [ 1.       0.       0.       0.       0.     ]
 [ 1.       0.       0.       0.       0.     ]
 [ 1.       0.       0.       0.       0.     ]
 [ 1.       0.       0.       0.       0.     ]]
 
Labels:
 [[25 22 29 31 20]
 [23 42 16  9 10]
 [ 1 40 12 10 42]
 [38 23 20  5 31]
 [17 12  9 33 35]
 [18 40 26 37  1]
 [33  1 26 11 40]
 [13 35 15 28 39]]
