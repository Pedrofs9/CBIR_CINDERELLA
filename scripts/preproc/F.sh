#!/bin/bash
#SBATCH --partition=cpu_8cores 
#SBATCH --qos=cpu_8cores_ext
#SBATCH --job-name=dataproc_E
#SBATCH --output=dataproc_E.out
#SBATCH --error=dataproc_E.err


echo "CINDERELLA BreLoAI Retrieval: A Study with Attention Mechanisms"
echo "Catalogue Type: E"
python src/main_dataproc.py \
 --config_json 'config/dataproc/E.json' \
 --images_resized_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/resized_224_correct' \
 --images_original_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/breloai-rsz-v2_copy' \
 --csvs_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/csvs' \
 --pickles_path '/nas-ctm01/homes/pferreira/Cinderella_Pedro/pickles/E_correct'
echo "Finished"