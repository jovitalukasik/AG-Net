# Bash script to use the pretrained generative network on NAS-Bench-301 
# and search using our AG-Net in the TENAS setting
# here examplary on CIFAR 10 data
# Table 4 in the main paper

cifar_data_path='path to cifar10 data or Imagenet-1k'
image_data=cifar10

python Search_TENAS.py  --image_data $image_data --cifar_data_path $cifar_data_path --device 'cuda:0'