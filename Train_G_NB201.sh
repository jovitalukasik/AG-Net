# Bash script to train the generative network in the NAS-Bench-201 search space


dataset=NB201

python Training_Generator.py  --device 'cuda:0' --dataset $dataset --ticks 500 --tick_size 5_000


