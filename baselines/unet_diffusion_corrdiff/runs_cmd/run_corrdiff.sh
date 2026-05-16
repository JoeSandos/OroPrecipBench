
###############################
#        残差扩散模型          #
###############################

# part1_US  数据集 训练
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py --config-name=config_training_part1_US_Appalachian.yaml \
    regression_checkpoint_path="training-runs/regression/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py --config-name=config_training_part1_US_Cascades.yaml \
    regression_checkpoint_path="training-runs/regression/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py --config-name=config_training_part1_US_Great_Plains.yaml \
    regression_checkpoint_path="training-runs/regression/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py --config-name=config_training_part1_US_Nevada.yaml \
    regression_checkpoint_path="training-runs/regression/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py --config-name=config_training_part1_US_Rocky.yaml \
    regression_checkpoint_path="training-runs/regression/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/network-snapshot-000242.pkl"



# part1_US  数据集 推理
## 指定 checkpoint_dir 为训练好的模型权重路径, 以 .pkl 结尾
## 推理时, 指定 seed_batch_size, 要求 nproc-per-node(进程数) * seed_batch_size = num_ensembles


CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part1_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/network-snapshot-000242.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00000-MyDataset-part1-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000182.pkl" 

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part1_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/network-snapshot-000242.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00001-MyDataset-part1-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000182.pkl" 

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part1_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/network-snapshot-000242.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00002-MyDataset-part1-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000182.pkl" 

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part1_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/network-snapshot-000242.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00003-MyDataset-part1-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000182.pkl" 

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part1_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/network-snapshot-000242.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00004-MyDataset-part1-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000182.pkl" 

# part1_US ERA5 数据集 迁移 推理
## 指定 checkpoint_dir 为训练好的模型权重路径, 以 .pkl 结尾
## 推理时, 指定 seed_batch_size, 要求 nproc-per-node(进程数) * seed_batch_size = num_ensembles

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part1_US_Appalachian_ERA5.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/network-snapshot-000242.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00000-MyDataset-part1-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000182.pkl" 

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part1_US_Cascades_ERA5.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/network-snapshot-000242.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00001-MyDataset-part1-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000182.pkl" 

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part1_US_Great_Plains_ERA5.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/network-snapshot-000242.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00002-MyDataset-part1-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000182.pkl" 

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part1_US_Nevada_ERA5.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/network-snapshot-000242.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00003-MyDataset-part1-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000182.pkl" 

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part1_US_Rocky_ERA5.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/network-snapshot-000242.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00004-MyDataset-part1-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000182.pkl" 


# part2_US  数据集 训练
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py --config-name=config_training_part2_US_Appalachian.yaml \
    regression_checkpoint_path="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py --config-name=config_training_part2_US_Cascades.yaml \
    regression_checkpoint_path="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py --config-name=config_training_part2_US_Great_Plains.yaml \
    regression_checkpoint_path="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py --config-name=config_training_part2_US_Nevada.yaml \
    regression_checkpoint_path="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_residual_diffusion.py --config-name=config_training_part2_US_Rocky.yaml \
    regression_checkpoint_path="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl"



# part2_US  数据集 推理
## 指定 checkpoint_dir 为训练好的模型权重路径, 以 .pkl 结尾
## 推理时, 指定 seed_batch_size, 要求 nproc-per-node(进程数) * seed_batch_size = num_ensembles

# ============================================================
# Model: Appalachian regression + residual checkpoint -> infer on 5 configs
# ============================================================

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"


# ============================================================
# Model: Cascades regression + residual checkpoint -> infer on 5 configs
# ============================================================

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"


# ============================================================
# Model: Great_Plains regression + residual checkpoint -> infer on 5 configs
# ============================================================

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"


# ============================================================
# Model: Nevada regression + residual checkpoint -> infer on 5 configs
# ============================================================

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"


# ============================================================
# Model: Rocky regression + residual checkpoint -> infer on 5 configs
# ============================================================

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node 2 generate_residual_diffusion.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.regression_checkpoint_path="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl" \
    generation.residual_checkpoint_path="training-runs/residual_diffusions/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/network-snapshot-000242.pkl"

###### 评估 ######

# part1-US-Appalachian
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Appalachian \
    --pred-path generated/residual_diffusion/part1-US-Appalachian/test_data_in/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/00000-MyDataset-part1-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000242-res_000182.nc \
    --out-dir results/residual_diffusion/part1-US-Appalachian/test_data_in/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/00000-MyDataset-part1-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part1-US-Cascades
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Cascades \
    --pred-path generated/residual_diffusion/part1-US-Cascades/test_data_in/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/00001-MyDataset-part1-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000242-res_000182.nc \
    --out-dir results/residual_diffusion/part1-US-Cascades/test_data_in/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/00001-MyDataset-part1-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part1-US-Great_Plains
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Great_Plains \
    --pred-path generated/residual_diffusion/part1-US-Great_Plains/test_data_in/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/00002-MyDataset-part1-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000242-res_000182.nc \
    --out-dir results/residual_diffusion/part1-US-Great_Plains/test_data_in/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/00002-MyDataset-part1-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part1-US-Nevada
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Nevada \
    --pred-path generated/residual_diffusion/part1-US-Nevada/test_data_in/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/00003-MyDataset-part1-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000242-res_000182.nc \
    --out-dir results/residual_diffusion/part1-US-Nevada/test_data_in/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/00003-MyDataset-part1-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part1-US-Rocky
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Rocky \
    --pred-path generated/residual_diffusion/part1-US-Rocky/test_data_in/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/00004-MyDataset-part1-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000242-res_000182.nc \
    --out-dir results/residual_diffusion/part1-US-Rocky/test_data_in/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/00004-MyDataset-part1-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/

    
# part1_US ERA5 数据集 迁移 评估


# part1-US-Appalachian
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Appalachian \
    --pred-path generated/residual_diffusion/part1-US-Appalachian/test_data_in_ERA5/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/00000-MyDataset-part1-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000242-res_000182.nc \
    --out-dir results/residual_diffusion/part1-US-Appalachian/test_data_in_ERA5/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/00000-MyDataset-part1-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part1-US-Cascades
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Cascades \
    --pred-path generated/residual_diffusion/part1-US-Cascades/test_data_in_ERA5/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/00001-MyDataset-part1-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000242-res_000182.nc \
    --out-dir results/residual_diffusion/part1-US-Cascades/test_data_in_ERA5/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/00001-MyDataset-part1-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part1-US-Great_Plains
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Great_Plains \
    --pred-path generated/residual_diffusion/part1-US-Great_Plains/test_data_in_ERA5/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/00002-MyDataset-part1-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000242-res_000182.nc \
    --out-dir results/residual_diffusion/part1-US-Great_Plains/test_data_in_ERA5/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/00002-MyDataset-part1-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part1-US-Nevada
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Nevada \
    --pred-path generated/residual_diffusion/part1-US-Nevada/test_data_in_ERA5/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/00003-MyDataset-part1-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000242-res_000182.nc \
    --out-dir results/residual_diffusion/part1-US-Nevada/test_data_in_ERA5/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/00003-MyDataset-part1-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part1-US-Rocky
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Rocky \
    --pred-path generated/residual_diffusion/part1-US-Rocky/test_data_in_ERA5/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/00004-MyDataset-part1-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000242-res_000182.nc \
    --out-dir results/residual_diffusion/part1-US-Rocky/test_data_in_ERA5/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/00004-MyDataset-part1-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/


# part2_US 数据集 评估 5*5

# part2-US-Appalachian / 00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/residual_diffusion/part2-US-Appalachian/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Appalachian/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Appalachian / 00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/residual_diffusion/part2-US-Appalachian/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Appalachian/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Appalachian / 00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/residual_diffusion/part2-US-Appalachian/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Appalachian/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Appalachian / 00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/residual_diffusion/part2-US-Appalachian/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Appalachian/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Appalachian / 00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/residual_diffusion/part2-US-Appalachian/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Appalachian/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Cascades / 00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/residual_diffusion/part2-US-Cascades/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Cascades/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Cascades / 00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/residual_diffusion/part2-US-Cascades/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Cascades/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Cascades / 00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/residual_diffusion/part2-US-Cascades/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Cascades/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Cascades / 00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/residual_diffusion/part2-US-Cascades/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Cascades/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Cascades / 00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/residual_diffusion/part2-US-Cascades/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Cascades/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Great_Plains / 00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/residual_diffusion/part2-US-Great_Plains/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Great_Plains/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Great_Plains / 00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/residual_diffusion/part2-US-Great_Plains/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Great_Plains/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Great_Plains / 00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/residual_diffusion/part2-US-Great_Plains/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Great_Plains/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Great_Plains / 00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/residual_diffusion/part2-US-Great_Plains/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Great_Plains/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Great_Plains / 00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/residual_diffusion/part2-US-Great_Plains/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Great_Plains/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Nevada / 00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/residual_diffusion/part2-US-Nevada/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Nevada/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Nevada / 00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/residual_diffusion/part2-US-Nevada/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Nevada/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Nevada / 00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/residual_diffusion/part2-US-Nevada/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Nevada/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Nevada / 00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/residual_diffusion/part2-US-Nevada/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Nevada/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Nevada / 00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/residual_diffusion/part2-US-Nevada/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Nevada/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Rocky / 00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/residual_diffusion/part2-US-Rocky/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Rocky/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/00008-MyDataset-part2-US-Great_Plains-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Rocky / 00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/residual_diffusion/part2-US-Rocky/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Rocky/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/00009-MyDataset-part2-US-Nevada-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Rocky / 00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/residual_diffusion/part2-US-Rocky/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Rocky/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/00010-MyDataset-part2-US-Rocky-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Rocky / 00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/residual_diffusion/part2-US-Rocky/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Rocky/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/00006-MyDataset-part2-US-Appalachian-ddpmpp-CorrDiff-gpus1-batch256-fp32/

# part2-US-Rocky / 00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/residual_diffusion/part2-US-Rocky/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/generation-ResDiff-reg_000465-res_000242.nc \
    --out-dir results/residual_diffusion/part2-US-Rocky/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/00007-MyDataset-part2-US-Cascades-ddpmpp-CorrDiff-gpus1-batch256-fp32/
