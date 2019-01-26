# Two Sigma Halite Challenge Code Repo

I developed a Gym environment for the Halite environment in order to train various reinforcement learning algorithms. For fast experimentation and testing these algorithms, I am using Ray a distributed computing library with the RLLib library which contains reinforcement learning components. Each of the algorithms in Ray can be scaled to a cluster. Therefore, I can perform distributed environment data collection and iterate faster to a solution.   

Some of the algorithms I am testing are:  
- DQN
- APEX DQN
- PPO

Additionally, I am testing out using Population Based Training in order to tune hyperparameters in one training cycle.  Also, I worked on creating an autoencoder for different board distributions of Halite, which can be found in the analysis directory. 


# Installation
1) Fill in necessary information in the ray-cluster.yaml file, this contains information such as what cloud provider you are using, what images to launch and how many to launch
2) Go to GCP and create the images for the Head and Worker, make sure to clone this repo and install the requirements.txt
3) `ray up ray-cluster.yaml`
4) Use the ssh command from the result of the last command
5) Launch one of the various algorithms and it will run on the distributed cluster you set up  

For more information on setup and the Ray environment checkout the [ray project](https://github.com/ray-project/ray).

# Authors
- Anthony Bisulco, arb426@cornell.edu
- Brian Wilcox, brianwilcox@stanford.edu
