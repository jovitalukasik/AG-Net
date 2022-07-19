# Bash script to use the pretrained generative network on NAS-Bench-101
# and search using our generator + XGB(XGB ranking)
# see Table 1 in the main paper

path=state_dicts/NASBench101
dataset=NB101


python Search_XGB.py --trials 10 --dataset $dataset --saved_path $path  --device 'cpu' --seed 1 --ticks 1 
python Search_XGBranking.py --trials 10 --dataset $dataset --saved_path $path  --device 'cpu' --seed 1 --ticks 1 
