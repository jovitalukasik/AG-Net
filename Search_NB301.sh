# Bash script to use the pretrained generative network on NAS-Bench-301
# and search using our AG-Net 

path=state_dicts/NASBench301
dataset=NB301


python Search_NB301.py --trials 10 --dataset $dataset --saved_path $path  --device 'cpu' --seed 1 --ticks 15