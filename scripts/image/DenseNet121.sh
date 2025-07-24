#!/bin/bash
#SBATCH --partition=gpu_min8gb
#SBATCH --qos=gpu_min8gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=/nas-ctm01/homes/pferreira/Cinderella_Pedro/results/DenseNet.out
#SBATCH --error=/nas-ctm01/homes/pferreira/Cinderella_Pedro/results/DenseNet.err


#VIS_DIR="/nas-ctm01/homes/pferreira/Cinderella_Pedro/visualizations/$(date +%Y-%m-%d_%H-%M-%S)"
#mkdir -p $VIS_DIR 
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
#export CUDA_VISIBLE_DEVICES=0  
#python src/main_image.py \
# --visualizations_path "$VIS_DIR" \
# --gpu_id 0 \
# --pickles_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/pickles/E' \
# --verbose \
# --train_or_test 'test' \
# --visualize_all \
# --generate_xai \
# --xai_backend MONAI \
# --xai_method SBSM \
# --max_visualizations 10000 \
# --results_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/results' \
# --checkpoint_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/results/2025-05-22_14-44-22' \
# --xai_batch_size 1 
#echo "Finished"

echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
 echo "Training Catalogue Type: E"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json 'config/image/E/DenseNet121.json' \
  --pickles_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/pickles/E' \
  --results_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/results' \
  --train_or_test 'train'
 echo "Finished"
#echo "Testing Catalogue Type: E"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/pickles/E' \
# --train_or_test 'test' \
# --checkpoint_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/results/2025-03-31_12-37-04' \
# --verbose
#echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/F/DaViT_Base.json' \
#  --pickles_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/pickles/F' \
#  --results_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/results' \
#  --train_or_test 'train'
# echo "Finished"
#echo "Testing Catalogue Type: F"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
# --train_or_test 'test' \
# --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_02-20-44/' \
# --verbose
#echo "Finished"