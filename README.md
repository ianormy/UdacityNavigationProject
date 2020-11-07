[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Navigation Project
### Introduction

This is my Navigation project for the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

For this project I trained an agent to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent was required to get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in this GitHub repository and unzip (or decompress) the file. 

## Setup
All the training was done on a Udacity Workspace. In order to do your own training using my [Training Jupyter Notebook](Training.ipynb) you will need to run it in the same environment. The workspace uses an old version of the Unity ML-Agents - v0.4 - that includes the BananaCollecter brain.

### Local Setup
In order to see my [Results](Results.ipynb), [Report](Report.ipnyb) and discussion about [Deep Q-Learning](Deep_Q-Learning.ipynb) you need only install the following packages from PyPi:

* numpy
* matplotlib
* jupyter

## Project Files
I initially trained the agent using DQN and saved the training results in the file [dqn_results.npy](dqn_results.npy). I then used a Double DQN (DDQN) and found it did a lot better. I saved the training results in the file [ddqn_results.npy](ddqn_results.npy). Both these files have been included in this repository. To see the results of this training which includes a comparison of the two different methods, please look at [Results](Results.ipynb).

## Training the Agent
If you want to train your own agent then use the [Training Jupyter Notebook](Training.ipynb) from within the Udacity Workspace. This will train the model using DDQN and save it to the file [ddqn_checkpoint.pth](ddqn_checkpoint.pth). It will also save the training results to in the file [ddqn_results.npy](ddqn_results.npy). I have included these files in this repository.
