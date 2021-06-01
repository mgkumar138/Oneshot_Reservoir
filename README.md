# One-shot learning of paired associations by a reservoir computing model with Hebbian plasticity

This repository is the official implementation of One-shot learning of paired associations by a reservoir computing model with Hebbian plasticity. 

The main result of the paper is to demonstrate the one-shot learning of multiple target coordinates using a Reservoir computing model in a single displaced location and multiple paired association navigation task.

4 agents were evaluated in both tasks and script begins with the following nomenclature:
Advantage Actor Critic (A2C)                 - a2c*
Symbolic agent                               - sym_mc*
Reservoir agent trained by perceptron rule   - res*
Reservoir trained by sparse learning signal  - sl_res_mc*


## Requirements

Refer to list of dependencies used in requirements.txt

Main dependency is Tensorflow 2.3.0

To install requirements:

```setup
pip install -r requirements.txt
```


## Single displaced locationn training & evaluation

To run the 4 types of agent(s) described in the paper in the single displaced location task, set working directory to ./single_reward and run this command:

```train
python [a2c_1pa/sym_mc_1pa/res_1pa_mc/sl_res_mc_1pa].py
```


## Multiple paired association training & evaluation

To run the 4 types of agent(s) described in the paper in the multiple paired association task, set working directory to ./multiple_pa and run this command:

```train
python [a2c_6pa/sym_mc_6pa/res_6pa_mc/sl_res_mc_6pa].py
```

## One-shot learning of 12 novel paired associates 
To run the 4 types of agent(s) described in the paper in the 12NPA task, set working directory to ./other_pa and run this command:

```train
python [a2c_12pa/sym_mc_12npa/res_nmc_12pa_mc/sl_res_nmc_12pa].py
```

Hyperparameters are set to obtain results in paper and can be tuned in the respective *.py scripts.
```

## Training details

Since the outcome of the paper is to demonstrate one-shot learning, there are no pretrained models. Instead, the learning potential of each agent can be observed by running the respective scripts.
Training for each agent takes about 15 minutes for single reward task and 30 minutes for multiple paired association task. Some agents would take a shorter time. 


## Results

Our agents achieve the following performance for single displaced location task :

Latency reached by all agents:
![Latency_1pa](https://user-images.githubusercontent.com/35286288/120262629-9c84d500-c2cc-11eb-9bdf-0823201c7160.png)

Time spent at each location during probe trial:
![Dgr_1pa](https://user-images.githubusercontent.com/35286288/120262683-bc1bfd80-c2cc-11eb-943d-3e4e4997b6f5.png)


Our agents achieve the following performance when learning multiple paired assocationn task :
Latency reached by all agents:
![Latency_6pa](https://user-images.githubusercontent.com/35286288/120262703-cb02b000-c2cc-11eb-8369-8bf375020f17.png)

Average visit ratio at during each probe session:
![Dgr_train_6pa](https://user-images.githubusercontent.com/35286288/120262752-e4a3f780-c2cc-11eb-9275-50d246fcdcc8.png)

One shot learning results obtained for session 22 (OPA), 24 (2NPA), 26 (6NPA)
![Dgr_eval_6pa](https://user-images.githubusercontent.com/35286288/120262911-3e0c2680-c2cd-11eb-931a-304c1567b800.png)

One shot learning results for 12 random paired assocations with varying Reservoir size
![PI_12pa_se](https://user-images.githubusercontent.com/35286288/120262953-511ef680-c2cd-11eb-8910-fd51bce9f6fa.png)

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
