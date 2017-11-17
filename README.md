 ## Project Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In order to extract the HOG features I used the hog function from the skimage.feature package.

The code for this step is contained in the third code cell of the following IPython notebook, under the *get_hog_features* function:

<i class="icon-file"></i> [vehicleDetection.ipnb](./vehicleDetection.ipnb)  

#### 2. Explain how you settled on your final choice of HOG parameters.

I then explored different color spaces and different `skimage.hog()` parameters (*color_space, orient, pixels_per_cell, cells_per_block*), and checked there outputs using images from the provided data set to get a feel for what the `skimage.hog()` output looks like. Those are the parameters I finally selected:

````
color_space = 'YCrCb' 
orient = 9  
pix_per_cell = 8 
cell_per_block = 2 
hog_channel = "ALL" 
````

Here is, as an example, the result for the final parameters on both car and not car image chosen randomly:

![HOG_IMAGE][./writeup/hog_features.png]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As recommended in the project, I chose Linear SVC.
To feed the classifier, I brought together spatial, histogram and HOG features, and normalized it.
Then using the full image data set, I trained it.
After trying several parameters combinations, with the selected one I was able to reach once over 99% of successful classification, and always well above 98%. This variation is normal considering that the training set and the test set are always redefined at the beginning of the training.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Looking at the test images we can easily see a pattern of what a car is on the images. With that in mind,  a full search over the complete image is useless and time consuming, once cars with a particular size can only be on a limited area of the image. Since there is no cars in the sky (or not yet :) ), at least the top part of the image can be discarded. Then, because the camera is on the top of the car, and it is pointed to the horizon, an acceptable approximation is that all the cars will be leveled by a line that goes around their top. Besides that, the closer the car are the bigger they look on the image.

So my idea was to consider single lines of different windows size leveled (roughly, since I just change it in few pixels). That way, thinner lines would detect smaller/farther cars while thicker lines would detect bigger/closer cars.

So I started testing my idea using lines with 256 pixels high down to 32 px by steps of 32 px, and started getting a better picture of what which could achieve.

Unfortunately, for the sake of processing speed I could not afford to use all of them, so I filtered it based on the results achieved and the above tests and finally came to this windows layout:

- 1 line of 160 px high starting at line y=400
- 1 line of 96 px high starting at line y=404

I also used a 75% overlay of windows on both lines, since it managed to have a better distinguish false detections (that have a more random behavior) from car detections.

That solution allowed me to get get good enough results that could be bettered with post processing.

I used the recommended single HOG feature extraction to make the system faster, so instead of resizing the windows to the training size of 64x64 I did a single resize to the full image using a scaling factor, that in other words will have the same effect but will be achieved in the opposite path.

The code is visible between lines 20 and 40 on the 10th cell, and from line 25 of the 7th cell (find_cars_in_image funtion) from this notebook: 

<i class="icon-file"></i> [vehicleDetection.ipnb](./vehicleDetection.ipnb)  

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![DETECTION_IMAGE][./writeup/pipeline.png]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_image/output_project_video_completed.mp4)

The thin yellow boxes show the detections out of the classifier. There we can see that some parts of the image can fool the classifier (like guards, mainly metalic ones over bridges, as well as dark traces over the whiter pavement), but since it mostly happens sporadically, using a valid filtering post-processing I managed to discard the majority of those false positives.

The thick red boxes are the tracking results out of the post-processing using spatial and temporal filtering.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Spatial filtering is used on each frame. The trick was to use a threshold level dependent of the detections (the heatmap had bigger values over the cars with the...). That allowed to achieve a very balanced filtering (avoiding to remove cars or keep to much false positives).

On the temporal side I considered the last 10 frames, that corresponds to 0.4 seconds on a video of 25 fps. I defined an object class to help storing detections. Then it was easy to retrieve and update them whenever they were considered as being a matching detection from the previous frames.

>I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

>Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most tricky parts (besides getting a bug free code when we are working late at night :-) ) are getting a light windows search and an efficient post processing.

It would have been easy to avoid looking to the left side of the image, but that would create a failure if we move the car to the right lanes (where it should have always been based, at least, on European laws :) ).

An improvement would be to do refined searches on limited small areas of the image where it would be expected to have a car based on the previous images. For that I added a variable (move) on the detection object to store the motion vector calculated from the detection centers from all the consecutive matches.  

In-line with what is required to stand out, this project was not just about doing a good detection, but also on doing it quite fast. Abusing on sliding windows would make it easier to get a better detection, but that would come at a price that would not justify it anymore.

As explained earlier, finding a good balance between detections and speed was the key here. My final choice was using one single line of windows with 160x160 pixels together with another line of 96x96 windows, both with a horizontal overlaping of 75%. That creates 156 windows, and seaching vehicles on them took a total processing time of 8.43 minutes for a total of 1260 frames of the video. That gives us an average of around 4.5 fps, which is not bad for a python code.
