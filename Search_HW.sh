# Exemplary Search on Hardware-Aware NASBench for the target device "Edge GPU" 
# with latency constraint L=5
# on ImageNet16-120
# Joint=0 for Eq. 5 in Paper
# Joint=1 for Eq. 4 in Paper

path=state_dicts/NASBench201
dataset=NB201
target_device=edgegpu
max_latency=5
image_data=ImageNet16-120

joint=0
python Search_Tabular.py --trials 1 --dataset $dataset --saved_path $path  --device 'cpu' --seed 1 --ticks 30 --image_data $image_data --target_device $target_device --max_latency $max_latency --joint $joint


joint=1
python Search_Tabular.py --trials 1 --dataset $dataset --saved_path $path  --device 'cpu' --seed 1 --ticks 30 --image_data $image_data --target_device $target_device --max_latency $max_latency --joint $joint
