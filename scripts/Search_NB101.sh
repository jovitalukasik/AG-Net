# Bash script to use the pretrained generative network on NAS-Bench-101
# and search using our AG-Net

path=state_dicts/NASBench101
dataset=NB101


python Search_Tabular.py --trials 10 --dataset $dataset --saved_path $path  --device 'cpu' --seed 1 --ticks 15