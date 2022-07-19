# Bash script to use the pretrained generative network on NAS-Bench-101
# and search using our AG-Net for ablation studies
# See Table 6 in the main paper

path=state_dicts/NASBench101
dataset=NB101


python Search_Tabular_wo_LSO.py --trials 1 --dataset $dataset --saved_path $path  --device 'cpu' --seed 0 --ticks 15 
python Search_Tabular_wo_BP.py --trials 1 --dataset $dataset --saved_path $path  --device 'cpu' --seed 0 --ticks 15 
