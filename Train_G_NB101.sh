# Bash script to train the generative network in the NAS-Bench-101 search space

dataset=NB101

python Training_Generator.py  --device 'cuda:0' --dataset $dataset --ticks 500 --tick_size 5_000


