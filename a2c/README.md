# Tuned A2C

## Major Changes
- Number of actors increased to 256
- Rollout length increased from 5 to 20
- Loss is averaged across batches instead of summed
- Gradient clipping decreased to 0.5
- RMSprop epsilon decreased to 1e-8

## A2C
A2C using the original model architecture as described in https://arxiv.org/abs/1602.01783

## A2C Small
A2C using a modified model architecture tuned for smaller input as described in https://bcourses.berkeley.edu/files/70573736/download?download_frd=1.

Achieves similar performance with a 1.84x speedup.
