Self-Driving Behavioral Cloning
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Data collection

The driving simulator is pretty straightforward to use. But you still probably want to do a couple of practice runs around the tracks before you hit the recording button. When recording, you might want to divide the track into smaller segments if you run into memory issues and/or saving image files. 

### Model design and training

Here we need to design a model that predicts a steering angle. Since Keras is used for the project, it is really easy to experiment with different modesl. I started with a simple model with no convolutional layers. After a few epochs of training, it became obvious pretty quickly that this is not going to work.

Because of the publicity and wide availability of NVidia's end-to-end driving paper, I immediately pivoted to use the NVidia model, which has five conv layers and three fully connected layers as shown in this [architecture diagram](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture.png)

#### Data augmentation

#### Data generator
