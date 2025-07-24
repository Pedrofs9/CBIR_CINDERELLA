#!/bin/bash
#SBATCH --partition=gpu_min11gb
#SBATCH --qos=gpu_min11gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=/nas-ctm01/homes/pferreira/Cinderella_Pedro/results/DeiT_Base_Patch16_224.out
#SBATCH --error=/nas-ctm01/homes/pferreira/Cinderella_Pedro/results/DeiT_Base_Patch16_224.err



echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
 echo "Training Catalogue Type: E"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json '/nas-ctm01/homes/pferreira/Cinderella_Pedro/config/image/E/DeiT_Base_Patch16_224.json' \
  --pickles_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/pickles/E_segmented' \
  --results_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/results' \
  --train_or_test 'train'
 echo "Finished"
#echo "Testing Catalogue Type: E"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
# --train_or_test 'test' \
# --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-17_04-23-52/' \
# --verbose
#echo "Finished"

# echo "Training Catalogue Type: F"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json 'config/image/F/DeiT_Base_Patch16_224.json' \
#  --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
#  --results_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F' \
#  --train_or_test 'train'
# echo "Finished"
#echo "Testing Catalogue Type: F"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
# --train_or_test 'test' \
#--checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_10-17-31/' \
# --verbose
#echo "Finished"