# Bash script to train the generative network in the NAS-Bench-NLP search space


dataset=NBNLP

python Training_Generator.py  --device 'cuda:0' --dataset $dataset --ticks 500 --tick_size 5_000


