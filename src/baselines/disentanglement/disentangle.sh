#!/bin/bash
#SBATCH --time=24:00:00        
#SBATCH --nodes 1  
#SBATCH --gpus-per-node=1      
#SBATCH --mem=64G                
#SBATCH --output=./logs/dialBERT.out
#SBATCH --job-name=dialBERT
#SBATCH --account=rrg-mrdal22			      
#SBATCH --mail-user=usmi.mukherjee@dal.ca
#SBATCH --mail-type=ALL

module load python/3.10
source venv3.10/bin/activate
python ./BugMentor/src/disentangle/5__disentangle.py
module unload python/3.10
deactivate