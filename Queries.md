### Preprocessing --> ML1M, Yahoo, Pinterest Done
sbatch --job-name=LXRpreproc --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/data_processing.py"

### Recommender Training
sbatch --job-name=CE4RS --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/recommenders_training.py"

sbatch --job-name=XXXX --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --trial 0 --learning_rate 0.004 --lambda_neg XXX --lambda_pos XXX --alpha XXX"


sbatch --job-name=Pos --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --trial 0 --learning_rate 0.004 --lambda_neg 1 --lambda_pos 35 --alpha 1"

sbatch --job-name=Neg --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --trial 0 --learning_rate 0.004 --lambda_neg 35 --lambda_pos 1 --alpha 1"

sbatch --job-name=Equal --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --trial 0 --learning_rate 0.004 --lambda_neg 35 --lambda_pos 35 --alpha 1"

sbatch --job-name=Alpha --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --trial 0 --learning_rate 0.004 --lambda_neg 8.3 --lambda_pos 35 --alpha 20"


## LXR Training  
sbatch --job-name=LXR --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/LXR_training.py --directory ML1M/MF/Recommenders/MLP_ML1M_0_512.pt --whereToSave ML1M/MF/LXR/0 --model MLP --data ML1M"

--> Directory: Recommender Checkpoint --> Ml1M/Recommenders/VAE_ML1M_1_12_256.pt
--> whereToSave: Where to dump the LXR checkpoint --> ML1M/LXR/number
--> model: VAE, MLP
--> data: ML1M. Pinterest, Yahoo


 


## Metrics
python code/metrics.py --directoryRec Sel1/VAE_ML1M_2_0_64.pt --directoryLXR LXR1/0/LXR_ML1M_VAE_0_24_128_35_7.pt --SizeLXR 128

### For xp size (best Rec & LXR) (Sparsity)
python code/metricsXpSize.py --directoryRec Sel1/VAE_ML1M_2_18_64.pt --directoryLXR LXR1/18/LXR_ML1M_VAE_1_24_128_35_7.pt --SizeLXR 128 --sizeXP 10

1
sbatch --job-name=LXR --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/metricsXpSize.py --directoryRec Sel1/VAE_ML1M_2_18_64.pt --directoryLXR LXR1/18/LXR_ML1M_VAE_1_24_128_35_7.pt --SizeLXR 128 --sizeXP 1"

#### Pinterest Yahoo
sbatch --job-name=LXR --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/metricsXpSize.py --dataset Pinterest --sizeXP 1"



### For Rec TopK (Consistency)
python code/metricsTopK.py --directoryRec Sel1/VAE_ML1M_2_18_64.pt --directoryLXR LXR1/18/LXR_ML1M_VAE_1_24_128_35_7.pt --SizeLXR 128 --sizeXP 10 --lentopk 4

2
sbatch --job-name=LXR --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/metricsTopK.py --directoryRec Sel2/VAE_ML1M_1_2_256.pt --directoryLXR LXR2/2/LXR_ML1M_VAE_0_39_128_35_7.pt --SizeLXR 128 --sizeXP 10 --lentopk 1"



Fur 7,1 ToDo
3
<!-- sbatch --job-name=LXR --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/metricsTopK.py --directoryRec Sel1/VAE_ML1M_2_7_64.pt --directoryLXR LXR1/7/LXR_ML1M_VAE_1_24_128_35_7.pt --SizeLXR 128 --sizeXP 10 --lentopk 3"
sbatch --job-name=LXR --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/metricsTopK.py --directoryRec Sel1/VAE_ML1M_2_1_64.pt --directoryLXR LXR1/1/LXR_ML1M_VAE_1_24_128_35_7.pt --SizeLXR 128 --sizeXP 10 --lentopk 3" -->

4
<!-- sbatch --job-name=LXR --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/metricsTopK.py --directoryRec Sel1/VAE_ML1M_2_7_64.pt --directoryLXR LXR1/7/LXR_ML1M_VAE_1_24_128_35_7.pt --SizeLXR 128 --sizeXP 10 --lentopk 4" -->
sbatch --job-name=LXR --mail-user=amir.reza@uibk.ac.at --time=12:00:00 --mem=64G ~/jobs/single-node-gpu.job "python code/metricsTopK.py --directoryRec Sel1/VAE_ML1M_2_1_64.pt --directoryLXR LXR1/1/LXR_ML1M_VAE_1_24_128_35_7.pt --SizeLXR 128 --sizeXP 10 --lentopk 4"




### commands
srun --nodes=1 --nodelist=gc7 --ntasks-per-node=1 --time=12:00:00 --pty bash -i
source ../CFX-REC/cfx-recVenv/bin/activate