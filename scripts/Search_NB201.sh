# Bash script to use the pretrained generative network on NAS-Bench-201
# and search using our AG-Net

path=state_dicts/NASBench201
dataset=NB201

#Cifar 10
image_data=cifar10_valid_converged
python Search_Tabular.py --trials 10 --dataset $dataset --saved_path $path  --device 'cpu' --seed 1 --ticks 30 --image_data $image_data

#Cifar 100
image_data=cifar100
python Search_Tabular.py --trials 10 --dataset $dataset --saved_path $path  --device 'cpu' --seed 1 --ticks 30 --image_data $image_data

#ImageNet16-120
image_data=ImageNet16-120
python Search_Tabular.py --trials 10 --dataset $dataset --saved_path $path  --device 'cpu' --seed 1 --ticks 30 --image_data $image_data

