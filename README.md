# Self-Driving-Behavior-Cloning
Using Convolutional Neural Network to clone self-driving behavior with drive simulator data

The basic idea is that you collect driving data of a car simulator operated by human. Then these data, including images captured by simulated camera mounted on the hood and steering angles, can be feed to a Convolutional Neural Network to learn how humans drive. The trained model can be later used to drive the simulator. For more details on how the model is constructed and trained, see this file [project_writeup.md](https://github.com/yadongliu/Self-Driving-Behavior-Cloning/blob/master/project_writeup.md)

## How to run the project for yourself

Get the [Udacity Car Simulator](https://github.com/udacity/self-driving-car-sim)

Start the Car Simulator and press 'R' to start recording some driving data on the built-in track.

Now you want to train the model on your own data, run: 
```sh
python model.py
```
On an AWS g2large instance, each epoch takes under a minute to run. 

If your training data set is good, you will see the loss decrease and stablize, a pre-trained model file .h5 will be generated. To see how it performs, run the following:
```sh
python drive.py model.h5 [IMGFOLDER]
```
Then start the Car Simulator and select Track 1 and Autonomous Mode. The optional argument IMGFOLDER is where you want the Simulator to dump driving images to. You can later turn these images into a mp4 video by:
```sh
python video.py IMGFOLDER
```

