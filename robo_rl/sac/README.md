[Soft Actor Critic](https://arxiv.org/pdf/1801.01290.pdf)
 1. Learning rate and optimiser sensitivity. [Refer this](https://arxiv.org/pdf/1810.02525.pdf)
2. Reward scale is the most important hyperparameter
3. Learning rate schedule?
4. SAC + unbiased HER
- [x] SAC + TD3
5. [Reading for change of variable technique used for squashing](https://www.stat.washington.edu/~nehemyl/files/UW_MATH-    STAT395_functions-random-variables.pdf)
6. [Smoothed Action Value Functions for Learning Gaussian Policies](https://arxiv.org/pdf/1803.02348.pdf) 


Could successfully learn Humanoid in around 8000-8200 episodes. Got around 4000-5000 reward. Matches the results in paper

## Notes from rlkit

1. bias init default to 1
2. weight_init uniform for last layer 1e-3, and fanin_init for hidden 
3. mean. std weight regularisation 1e-3
4. Bigger tau 1e-2


