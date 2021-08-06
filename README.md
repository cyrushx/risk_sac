# Risk Conditioned Neural Motion Planning
Code of paper [Risk Conditioned Neural Motion Planning](https://arxiv.org/abs/2108.01851)

### Citation
```
@inproceedings{huang2021risksac,
  title={Risk Conditioned Neural Motion Planning},
  author={Huang, Xin and Feng, Meng and Jasour, Ashkan and Rosman, Guy and Williams, Brian C},
  booktitle={2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2021},
  organization={IEEE}
}
```

## Installation

* Install the Conda environment: follow instructions for Miniconda from the
[website](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

* Install Conda environment.

```bash
cd <repo-location>
conda env create -f env.yml
conda activate risk_sac
pip install -e .
```
NOTE: If `conda activate` does not work, replace with `source activate`.

* Add the repo directory to PYTHONPATH.

## RL Agent Implementation
Modify in RLKit submodule.

### Pull submodule after cloning this repo
```
git submodule add git@github.com:cyrushx/rlkit.git external/rlkit
git submodule update --init --recursive
git submodule update --remote --merge
```

### Install rlkit library
```bash
cd external/rlkit
pip install -e .
```

## Additional Comments:
Mujoco_py is not required, if see mujoco_py error, comment out following lines in 
risk_deeprl/external/rlkit/rlkit/envs/wrappers/__init__.py

```python
from rlkit.envs.wrappers.image_mujoco_env import ImageMujocoEnv
from rlkit.envs.wrappers.image_mujoco_env_with_obs import ImageMujocoWithObsEnv
```
---
## Train a standard SAC model in simple maze
### Train model
```sh
python model/train_sac.py
```

### Visualize training stats
```sh
python model/plot_learning_stats.py -i external/rlkit/data/YOUR_MODEL_PATH/
```

---
## Train a risk-bounded model in simple maze
### Train a model with fixed upper risk bound of 0.2
```sh
python model/train_risk_sac.py
```

### Visualize training stats
```sh
python utils/plot_learning_stats.py -i external/rlkit/data/YOUR_MODEL_PATH/
```

---
## Train risk-conditioned model in simple maze
### Train a model
```sh
python model/train_risk_conditioned_sac.py --delta 0.2 --risk-coeff 10
```

### Visualize stats
```sh
python utils/plot_learning_stats.py -i external/rlkit/data/YOUR_MODEL_PATH/
```

### Visualize paths with baseline paths
```sh
python external/rlkit/scripts/run_policy.py external/rlkit/data/YOUR_MODEL_PATH/params.pkl -v --baseline
```

---
## Train risk-conditioned model in FlyTrapBig maze
### Train model
```sh
python model/train_risk_conditioned_sac.py --env FlyTrapBig --risk-coeff 20 --epochs 1200
```

### Visualize paths with different risk bounds at a fixed starting location
```sh
python external/rlkit/scripts/run_policy.py external/rlkit/data/YOUR_MODEL_PATH/params.pkl -v
```

### Visualize paths with the same risk bound at different starting locations
```sh
python external/rlkit/scripts/run_policy.py external/rlkit/data/YOUR_MODEL_PATH/params.pkl -v --multiple-start
```

---
## Train risk-conditioned model with nonlinear Dubins dynamics
### Train model
```sh
python model/train_risk_conditioned_sac.py --env TwoRooms --delta 0.1 --risk-coeff 20 --epochs 500 --dubins
```

### Visualize paths
```sh
python external/rlkit/scripts/run_policy.py external/rlkit/data/YOUR_MODEL_PATH/params.pkl -v
```
