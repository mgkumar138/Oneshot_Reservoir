# One-shot learning of paired associations by a reservoir computing model with Hebbian plasticity

This repository is the implementation of One-shot learning of paired associations by a reservoir computing model with Hebbian plasticity. 

The main result of the paper is to demonstrate the one-shot learning of multiple target coordinates using a Reservoir computing model in a single displaced location and multiple paired association navigation task.

4 agents were evaluated in both tasks and script begins with the following nomenclature:
Advantage Actor Critic (A2C)                                            - a2c*
Hybrid temporal difference - symbolic agent                             - sym_mc*
Reservoir agent trained by perceptron rule                              - res_mc*
Reservoir trained by 4-factor variant of exploratory-Hebbian (EH) rule  - sl_res_mc*


## Requirements

System information

OS: Windows 10

Python distribution: Anaconda3-2021.05

Tensorflow == 2.3.0

```train
pip install tensorflow==2.3.0
```

## Single displaced locationn training & evaluation

To run the 4 types of agent(s) described in the paper in the single displaced location task, set working directory to ./single_reward and run this command:

```train
python [a2c_1pa/sym_mc_1pa/res_mc_1pa/sl_res_mc_1pa].py
```


## Multiple paired association training & evaluation

To run the 4 types of agent(s) described in the paper in the multiple paired association task, set working directory to ./multiple_pa and run this command:

```train
python [a2c_6pa/sym_mc_6pa/res_mc_6pa/sl_res_mc_6pa].py
```


## One-shot learning of 12 novel paired associates 
To run the 4 types of agent(s) described in the paper in the 12NPA task, set working directory to ./other_pa and run this command:

```train
python [a2c_12pa/sym_mc_12pa/res_mc_12pa/sl_res_mc_12pa].py
```

Hyperparameters are set to obtain results in paper and can be tuned in the respective *.py scripts.


## Training details

Since the outcome of the paper is to demonstrate one-shot learning, there are no pretrained models except for the neural motor controller. The learning potential of each agent can be observed by running the respective scripts.
Training for each agent takes about 15 minutes for single reward task and 30 minutes for multiple paired association task. Some agents would take a shorter time.

Most of the agent hyperparameters can be found in get_default_hp function in ./backend_scripts/utils.py. Critical hyperparameters are found in each *.py script.

E.g. if you would want the sym_/res_/sl_res*.py agents to use the symbolic or neural motor controller, set hp['usesmc'] to True or False respectively. 


## Results

Our agents achieve the following performance for single displaced location task :

Latency reached by all agents:

![Latency_1pa](https://user-images.githubusercontent.com/35286288/120445898-a76a6300-c3bb-11eb-8dd8-50068163b657.png)

Time spent at each location during probe trial:

![Dgr_1pa](https://user-images.githubusercontent.com/35286288/120445926-ad604400-c3bb-11eb-9add-251cd5e2fbdb.png)


Our agents achieve the following performance when learning multiple paired assocationn task :
Latency reached by all agents:

![Latency_6pa](https://user-images.githubusercontent.com/35286288/120445947-b224f800-c3bb-11eb-88a8-239e2e325099.png)

Average visit ratio at during each probe session:

![Dgr_train_6pa](https://user-images.githubusercontent.com/35286288/120445966-b94c0600-c3bb-11eb-9c6c-6a676cf70c4d.png)

One shot learning results obtained for session 22 (OPA), 24 (2NPA), 26 (6NPA)

![Dgr_eval_6pa](https://user-images.githubusercontent.com/35286288/120445974-bcdf8d00-c3bb-11eb-9159-abe9d18fc23d.png)

One shot learning results for 12 random paired assocations with varying Reservoir size

![PI_12pa_se](https://user-images.githubusercontent.com/35286288/120446029-c79a2200-c3bb-11eb-8f2d-b782f1a727ce.png)

## Contributing
Please cite the relevant work if the code is used for academic purposes.
