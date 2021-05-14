# intrinsics motivation experiments

# 0 testing environment

![testing_env](doc/env_tunnel.png)

**random network distillation**
![animation](doc/env_rnd.gif)


**random network distillation and entropy motivation**
![animation](doc/env_rnd_entropy.gif)


- the baseline agent is not able to solve environment
- the random network distillation can solve problem in arround 50% of cases
- random network distillation and entropy motivation solve problem in 100% cases 

![results](experiments/testing_env/results/score_per_iteration.png)

curiosity motivation for RND drop too quickly, and is not working as learning signal anymore
![results](experiments/testing_env/results/rnd_internal_motivation.png)


combination of curiosity and entropy can helps to learn
![results](experiments/testing_env/results/entropy_internal_motivation.png)

# dependences
cmake python3 python3-pip

**basic python libs**
pip3 install numpy matplotlib torch torchviz pillow opencv-python networkx

**environments**
pip3 install  gym pybullet pybulletgym 'gym[atari]' 'gym[box2d]' gym-super-mario-bros gym_2048