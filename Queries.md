sbatch --job-name=XXXX --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --trial 0 --learning_rate 0.004 --lambda_neg XXX --lambda_pos XXX --alpha XXX"


sbatch --job-name=Pos --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --trial 0 --learning_rate 0.004 --lambda_neg 1 --lambda_pos 35 --alpha 1"

sbatch --job-name=Neg --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --trial 0 --learning_rate 0.004 --lambda_neg 35 --lambda_pos 1 --alpha 1"

sbatch --job-name=Equal --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --trial 0 --learning_rate 0.004 --lambda_neg 35 --lambda_pos 35 --alpha 1"

sbatch --job-name=Alpha --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --trial 0 --learning_rate 0.004 --lambda_neg 8.3 --lambda_pos 35 --alpha 20"


## LXR Training  
sbatch --job-name=LXR --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --directory VAE_ML1M_2_18_64.pt --whereSaved LXR1/18"


## Metrics
python code/metrics.py --directoryRec Sel1/VAE_ML1M_2_0_64.pt --directoryLXR LXR1/0/LXR_ML1M_VAE_0_24_128_35_7.pt --SizeLXR 128

sbatch --job-name=LXR --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/metrics.py --directoryRec Sel1/VAE_ML1M_2_0_64.pt --directoryLXR LXR1/0/LXR_ML1M_VAE_0_24_128_35_7.pt --SizeLXR 128"



### commands
srun --nodes=1 --nodelist=gc7 --ntasks-per-node=1 --time=12:00:00 --pty bash -i
source ../CFX-REC/cfx-recVenv/bin/activate