#!/bin/bash
#SBATCH --partition=gpu_min24gb
#SBATCH --qos=gpu_min24gb_ext
#SBATCH --job-name=cind_breloai_att_ret
#SBATCH --output=/nas-ctm01/homes/pferreira/Cinderella_Pedro/results/Swin.out
#SBATCH --error=/nas-ctm01/homes/pferreira/Cinderella_Pedro/results/Swin.err



#echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
# echo "Training Catalogue Type: E"
# python src/main_image.py \
#  --gpu_id 0 \
#  --config_json '/nas-ctm01/homes/pferreira/Cinderella_Pedro/config/image/E/Swin_Base_Patch4_224.json' \
#  --pickles_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/pickles/E' \
#  --results_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/results' \
#  --train_or_test 'train'\
# echo "Finished"
#echo "Testing Catalogue Type: E"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/E' \
# --verbose \
# --train_or_test 'test' \
# --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/E/2024-11-16_22-11-38/'
#echo "Finished"

 echo "Training Catalogue Type: F"
 python src/main_image.py \
  --gpu_id 0 \
  --config_json '/nas-ctm01/homes/pferreira/Cinderella_Pedro/config/image/F/Swin_Base_Patch4_224.json' \
  --pickles_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/pickles/F' \
  --results_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/results' \
  --train_or_test 'train'
 echo "Finished"
#echo "Testing Catalogue Type: F"
#python src/main_image.py \
# --gpu_id 0 \
# --pickles_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/pickles/F' \
# --verbose \
# --train_or_test 'test' \
# --checkpoint_path '/nas-ctm01/datasets/private/CINDERELLA/experiments/retrieval/tgoncalv/results/F/2024-11-17_04-54-50/'
#echo "Finished"