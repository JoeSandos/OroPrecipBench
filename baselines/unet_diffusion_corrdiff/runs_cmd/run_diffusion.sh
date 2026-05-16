
###############################
#           扩散模型           #
###############################

# part1_US  数据集 训练
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part1_US_Appalachian.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part1_US_Cascades.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part1_US_Great_Plains.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part1_US_Nevada.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part1_US_Rocky.yaml


# part1_US  数据集 推理
## 指定 checkpoint_dir 为训练好的模型权重路径, 以 .pkl 结尾
## 推理时, 指定 seed_batch_size, 要求 nproc-per-node(进程数) * seed_batch_size = num_ensembles
CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part1_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00006-MyDataset-part1-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000303.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part1_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00004-MyDataset-part1-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000303.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part1_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00007-MyDataset-part1-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000303.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part1_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00005-MyDataset-part1-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000303.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part1_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00003-MyDataset-part1-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000303.pkl"





# part1_US ERA5 数据集 迁移 推理
## 指定 checkpoint_dir 为训练好的模型权重路径, 以 .pkl 结尾
## 推理时, 指定 seed_batch_size, 要求 nproc-per-node(进程数) * seed_batch_size = num_ensembles

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part1_US_Appalachian_ERA5.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00006-MyDataset-part1-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000303.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part1_US_Cascades_ERA5.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00004-MyDataset-part1-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000303.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part1_US_Great_Plains_ERA5.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00007-MyDataset-part1-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000303.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part1_US_Nevada_ERA5.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00005-MyDataset-part1-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000303.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part1_US_Rocky_ERA5.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00003-MyDataset-part1-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000303.pkl"




# part2_US  数据集 训练
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part2_US_Appalachian.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part2_US_Cascades.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part2_US_Great_Plains.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part2_US_Nevada.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_diffusion.py --config-name=config_training_part2_US_Rocky.yaml



# part2_US  数据集 推理GPU
# 耗时: 0.16 小时（4GPU）

## 指定 checkpoint_dir 为训练好的模型权重路径, 以 .pkl 结尾
## 推理时, 指定 seed_batch_size, 要求 nproc-per-node(进程数) * seed_batch_size = num_ensembles
# ============================================================
# Model: Appalachian checkpoint -> infer on 5 configs
# ============================================================

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"


# ============================================================
# Model: Cascades checkpoint -> infer on 5 configs
# ============================================================

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"


# ============================================================
# Model: Great_Plains checkpoint -> infer on 5 configs
# ============================================================

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"


# ============================================================
# Model: Nevada checkpoint -> infer on 5 configs
# ============================================================

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"


# ============================================================
# Model: Rocky checkpoint -> infer on 5 configs
# ============================================================

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"

CUDA_VISIBLE_DEVICES=5,6 torchrun --standalone --nproc-per-node 2 generate_diffusion.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.seed_batch_size=4 \
    generation.checkpoint_dir="training-runs/diffusion/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/network-snapshot-000202.pkl"




###### 评估 ######

# part1-US-Appalachian
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Appalachian \
    --pred-path generated/diffusion/part1-US-Appalachian/test_data_in/00006-MyDataset-part1-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000303.nc \
    --out-dir results/diffusion/part1-US-Appalachian/test_data_in/00006-MyDataset-part1-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part1-US-Cascades
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Cascades \
    --pred-path generated/diffusion/part1-US-Cascades/test_data_in/00004-MyDataset-part1-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000303.nc \
    --out-dir results/diffusion/part1-US-Cascades/test_data_in/00004-MyDataset-part1-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part1-US-Great_Plains
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Great_Plains \
    --pred-path generated/diffusion/part1-US-Great_Plains/test_data_in/00007-MyDataset-part1-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000303.nc \
    --out-dir results/diffusion/part1-US-Great_Plains/test_data_in/00007-MyDataset-part1-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part1-US-Nevada
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Nevada \
    --pred-path generated/diffusion/part1-US-Nevada/test_data_in/00005-MyDataset-part1-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000303.nc \
    --out-dir results/diffusion/part1-US-Nevada/test_data_in/00005-MyDataset-part1-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part1-US-Rocky
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Rocky \
    --pred-path generated/diffusion/part1-US-Rocky/test_data_in/00003-MyDataset-part1-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000303.nc \
    --out-dir results/diffusion/part1-US-Rocky/test_data_in/00003-MyDataset-part1-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part1_US ERA5 数据集 迁移 评估

# part1-US-Appalachian
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Appalachian \
    --pred-path generated/diffusion/part1-US-Appalachian/test_data_in_ERA5/00006-MyDataset-part1-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000303.nc \
    --out-dir results/diffusion/part1-US-Appalachian/test_data_in_ERA5/00006-MyDataset-part1-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part1-US-Cascades
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Cascades \
    --pred-path generated/diffusion/part1-US-Cascades/test_data_in_ERA5/00004-MyDataset-part1-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000303.nc \
    --out-dir results/diffusion/part1-US-Cascades/test_data_in_ERA5/00004-MyDataset-part1-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part1-US-Great_Plains
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Great_Plains \
    --pred-path generated/diffusion/part1-US-Great_Plains/test_data_in_ERA5/00007-MyDataset-part1-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000303.nc \
    --out-dir results/diffusion/part1-US-Great_Plains/test_data_in_ERA5/00007-MyDataset-part1-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part1-US-Nevada
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Nevada \
    --pred-path generated/diffusion/part1-US-Nevada/test_data_in_ERA5/00005-MyDataset-part1-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000303.nc \
    --out-dir results/diffusion/part1-US-Nevada/test_data_in_ERA5/00005-MyDataset-part1-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part1-US-Rocky
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Rocky \
    --pred-path generated/diffusion/part1-US-Rocky/test_data_in_ERA5/00003-MyDataset-part1-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000303.nc \
    --out-dir results/diffusion/part1-US-Rocky/test_data_in_ERA5/00003-MyDataset-part1-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US 评估 5*5

# part2-US-Appalachian / 00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/diffusion/part2-US-Appalachian/test_data_in/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Appalachian/test_data_in/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Appalachian / 00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/diffusion/part2-US-Appalachian/test_data_in/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Appalachian/test_data_in/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Appalachian / 00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/diffusion/part2-US-Appalachian/test_data_in/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Appalachian/test_data_in/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Appalachian / 00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/diffusion/part2-US-Appalachian/test_data_in/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Appalachian/test_data_in/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Appalachian / 00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/diffusion/part2-US-Appalachian/test_data_in/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Appalachian/test_data_in/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Cascades / 00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/diffusion/part2-US-Cascades/test_data_in/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Cascades/test_data_in/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Cascades / 00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/diffusion/part2-US-Cascades/test_data_in/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Cascades/test_data_in/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Cascades / 00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/diffusion/part2-US-Cascades/test_data_in/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Cascades/test_data_in/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Cascades / 00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/diffusion/part2-US-Cascades/test_data_in/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Cascades/test_data_in/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Cascades / 00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/diffusion/part2-US-Cascades/test_data_in/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Cascades/test_data_in/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Great_Plains / 00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/diffusion/part2-US-Great_Plains/test_data_in/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Great_Plains/test_data_in/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Great_Plains / 00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/diffusion/part2-US-Great_Plains/test_data_in/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Great_Plains/test_data_in/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Great_Plains / 00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/diffusion/part2-US-Great_Plains/test_data_in/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Great_Plains/test_data_in/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Great_Plains / 00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/diffusion/part2-US-Great_Plains/test_data_in/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Great_Plains/test_data_in/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Great_Plains / 00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/diffusion/part2-US-Great_Plains/test_data_in/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Great_Plains/test_data_in/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Nevada / 00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/diffusion/part2-US-Nevada/test_data_in/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Nevada/test_data_in/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Nevada / 00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/diffusion/part2-US-Nevada/test_data_in/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Nevada/test_data_in/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Nevada / 00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/diffusion/part2-US-Nevada/test_data_in/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Nevada/test_data_in/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Nevada / 00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/diffusion/part2-US-Nevada/test_data_in/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Nevada/test_data_in/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Nevada / 00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/diffusion/part2-US-Nevada/test_data_in/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Nevada/test_data_in/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Rocky / 00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/diffusion/part2-US-Rocky/test_data_in/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Rocky/test_data_in/00008-MyDataset-part2-US-Appalachian-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Rocky / 00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/diffusion/part2-US-Rocky/test_data_in/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Rocky/test_data_in/00009-MyDataset-part2-US-Cascades-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Rocky / 00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/diffusion/part2-US-Rocky/test_data_in/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Rocky/test_data_in/00010-MyDataset-part2-US-Great_Plains-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Rocky / 00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/diffusion/part2-US-Rocky/test_data_in/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Rocky/test_data_in/00011-MyDataset-part2-US-Nevada-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/

# part2-US-Rocky / 00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/diffusion/part2-US-Rocky/test_data_in/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/generation-network-snapshot-000202.nc \
    --out-dir results/diffusion/part2-US-Rocky/test_data_in/00012-MyDataset-part2-US-Rocky-ddpmpp-EDMPrecondSR-gpus1-batch256-fp32/
