# Post Training with Policy Gradients: Optimality and the Barrier of the Base Model

## Mathematical Setup

Our goal is to train an autoregressive model to predict a response sequence $\boldsymbol{y}$ to an input context $\boldsymbol{x}$, which is drawn from some distribution on $\mathbb{R}^d$. The response is \boldsymbol{y}=(y_1,...,y_N)$ where each $y_i$ is a number between $0$ and $k-1$. There are two setups of ground truth data generation that we consider:

1. Uniform $\boldsymbol{x}$: Here, $\boldsymbol{x}$ is drawn uniformly from the hypercube $\{\pm 1\}^d$. 

2. Gaussian Mixture Model $\boldsymbol{x}$: Here, $\boldsymbol{x}$ is drawn from a mixture of M isotropic Gaussians whose means are the first to $M$th standard basis of $\mathbb{R}^d$, and then its normalized.

In both cases, once $x$ is generated, the ground-truth response $y$ is generated as follows.
First, $y_1 = \argmax(\boldsymbol{W}^*_1 x)$ where $\boldsymbol{W}^*_1 \in \mathbb{R}^{k \times d}$. Then, $y_i = \argmax(\boldsymbol{W}^*_1 x + \boldsymbol{W}^*_2 \boldsymbol{e}_{y_{i-1}})$ for $i \geq 2$ where $\boldsymbol{e}_y$ is the $y$th standard basis of $\mathbb{R}^k$, and $\boldsymbol{W}^*_2 \in \mathbb{R}^{k \times k}$. We generate $\boldsymbol{W}^*_1$ and $\boldsymbol{W}^*_2$ randomly at first and then fix them.

Given $(\boldsymbol{x},y_1,...,y_i)$, the model predicts y_{i+1} by sampling it from 
```math
p_w(y|\boldsymbol{x},y_1,...,y_i) = \operatorname{softmax}(\boldsymbol{w}^\top \phi(\boldsymbol{x},y_1,...,y_i,y))
```
over $y \in \{0,...,N\}$. We define the feature map as
```math
\phi(x,y_1,...,y_i,y) = \operatorname{concat}(\operatorname{vec}(\boldsymbol{x}\boldsymbol{e}_y^\top), \operatorname{vec}(\boldsymbol{e}_{y}\boldsymbol{e}_{y_{i-1}}^\top)) \in \mathbb{R}^{dk + k^2}.
```

We want to consider three approaches. The first is to do supervised training with stochastic optimizers, where at each iteration we receive a batch of i.i.d. ground-truth data $(\boldsymbol{x}^i,\boldsymbol{y}^i)$.

The second approach is to train with outcome based rewards using policy gradients. Given a batch of contexts, we sample corresponding trajectories from current p_w. Assign a 1 or -1 reward based on the model being correct or not, and update the model with policy gradient on this batch.

In the third approach, we train with a process reward model. For any $i \in [N]$, define $r(x,x_1,...,y_i)$ to be 1 if the answer is correct so far, otherwise -1. We want to use this reward (possibly with mean subtraction baseline) directly as the advantage estimator of each step.

## Experiments

The following experiments are currently implemented:

- `sup_then_pg_outcome` (`configs/outcome_reward.yaml`): supervised pretraining followed by policy-gradient post-training with outcome-based rewards, with likelihood tracking.
- `sup_then_pg_process` (`configs/process_reward.yaml`): supervised pretraining followed by policy-gradient post-training with process rewards at each step, with likelihood tracking.
- `sup_cdf_quantile` (`configs/cdf_quantile.yaml`): supervised training with checkpoint evaluation using CDF, quantile, and alpha-tail plots.

## Run

Install `requirements.txt` in a new virtual environment. Then run one experiment per command (one config file per experiment type):

```bash
python3 main.py outcome_reward.yaml
```

Other experiment types:

```bash
python3 main.py process_reward.yaml
python3 main.py cdf_quantile.yaml
```

Artifacts are saved to `global.output_dir` from the config.
