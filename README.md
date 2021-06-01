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

Hyperparameters are set to obtain results in paper and can be tuned in the respective *.py scripts.


## Multiple paired association training & evaluation

To run the 4 types of agent(s) described in the paper in the multiple paired association task, set working directory to ./multiple_pa and run this command:

```train
python [a2c_6pa/sym_mc_6pa/res_6pa_mc/sl_res_mc_6pa].py
```

Hyperparameters are set to obtain results in paper and can be tuned in the respective *.py scripts.
```

## Training details

Since the outcome of the paper is to demonstrate one-shot learning, there are no pretrained models. Instead, the learning potential of each agent can be observed by running the respective scripts.
Training for each agent takes about 15 minutes for single reward task and 30 minutes for multiple paired association task. Some agents would take a shorter time. 


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
