# Bash script to use the pretrained generative network on NAS-Bench-NLP
# and search using our AG-Net

path=state_dicts/NASBenchNLP
dataset=NBNLP


python Search_NBNLP.py --trials 10 --dataset $dataset --saved_path $path  --device 'cpu' --seed 1 --ticks 30