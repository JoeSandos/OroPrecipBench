

###############################
#           回归模型           #
###############################

# part1_US  数据集 训练
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part1_US_Appalachian.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part1_US_Cascades.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part1_US_Great_Plains.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part1_US_Nevada.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part1_US_Rocky.yaml


# part1_US  数据集 推理
CUDA_VISIBLE_DEVICES=3 python generate_regression.py --config-name=config_generate_part1_US_Appalachian.yaml \
    generation.checkpoint_dir="training-runs/regression/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/network-snapshot-000242.pkl"
CUDA_VISIBLE_DEVICES=3 python generate_regression.py --config-name=config_generate_part1_US_Cascades.yaml \
    generation.checkpoint_dir="training-runs/regression/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/network-snapshot-000242.pkl"
CUDA_VISIBLE_DEVICES=4 python generate_regression.py --config-name=config_generate_part1_US_Great_Plains.yaml \
    generation.checkpoint_dir="training-runs/regression/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/network-snapshot-000242.pkl"
CUDA_VISIBLE_DEVICES=4 python generate_regression.py --config-name=config_generate_part1_US_Nevada.yaml \
    generation.checkpoint_dir="training-runs/regression/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/network-snapshot-000242.pkl"
CUDA_VISIBLE_DEVICES=7 python generate_regression.py --config-name=config_generate_part1_US_Rocky.yaml \
    generation.checkpoint_dir="training-runs/regression/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/network-snapshot-000242.pkl"


# part1_US ERA5 数据集 迁移 推理
CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part1_US_Appalachian_ERA5.yaml \
    generation.checkpoint_dir="training-runs/regression/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part1_US_Cascades_ERA5.yaml \
    generation.checkpoint_dir="training-runs/regression/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/network-snapshot-000242.pkl"
    
CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part1_US_Great_Plains_ERA5.yaml \
    generation.checkpoint_dir="training-runs/regression/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part1_US_Nevada_ERA5.yaml \
    generation.checkpoint_dir="training-runs/regression/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/network-snapshot-000242.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part1_US_Rocky_ERA5.yaml \
    generation.checkpoint_dir="training-runs/regression/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/network-snapshot-000242.pkl"



# part2_US  数据集 训练
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part2_US_Appalachian.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part2_US_Cascades.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part2_US_Great_Plains.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part2_US_Nevada.yaml
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 train_regression.py --config-name=config_training_part2_US_Rocky.yaml



# part2_US  数据集 推理


CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.checkpoint_dir="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.checkpoint_dir="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.checkpoint_dir="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.checkpoint_dir="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Appalachian.yaml \
    generation.checkpoint_dir="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl"



CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.checkpoint_dir="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.checkpoint_dir="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.checkpoint_dir="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.checkpoint_dir="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Cascades.yaml \
    generation.checkpoint_dir="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl"



CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.checkpoint_dir="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.checkpoint_dir="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.checkpoint_dir="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.checkpoint_dir="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Great_Plains.yaml \
    generation.checkpoint_dir="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl"



CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.checkpoint_dir="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.checkpoint_dir="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.checkpoint_dir="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.checkpoint_dir="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Nevada.yaml \
    generation.checkpoint_dir="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl"



CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.checkpoint_dir="training-runs/regression/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.checkpoint_dir="training-runs/regression/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.checkpoint_dir="training-runs/regression/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.checkpoint_dir="training-runs/regression/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/network-snapshot-000465.pkl"

CUDA_VISIBLE_DEVICES=0 python generate_regression.py --config-name=config_generate_part2_US_Rocky.yaml \
    generation.checkpoint_dir="training-runs/regression/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/network-snapshot-000465.pkl"


###### 评估 ######

# part1-US-Appalachian
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Appalachian \
    --pred-path generated/regression/part1-US-Appalachian/test_data_in/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/generation-network-snapshot-000242.nc \
    --out-dir results/regression/part1-US-Appalachian/test_data_in/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Appalachian \
    --pred-path generated/regression/part1-US-Appalachian/test_data_in_ERA5/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/generation-network-snapshot-000242.nc \
    --out-dir results/regression/part1-US-Appalachian/test_data_in_ERA5/00006-Regression-MyDataset-part1-US-Appalachian-ddpmpp-20260428_105936/

# part1-US-Cascades
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Cascades \
    --pred-path generated/regression/part1-US-Cascades/test_data_in/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/generation-network-snapshot-000242.nc \
    --out-dir results/regression/part1-US-Cascades/test_data_in/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Cascades \
    --pred-path generated/regression/part1-US-Cascades/test_data_in_ERA5/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/generation-network-snapshot-000242.nc \
    --out-dir results/regression/part1-US-Cascades/test_data_in_ERA5/00008-Regression-MyDataset-part1-US-Cascades-ddpmpp-20260428_195523/

# part1-US-Great_Plains
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Great_Plains \
    --pred-path generated/regression/part1-US-Great_Plains/test_data_in/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/generation-network-snapshot-000242.nc \
    --out-dir results/regression/part1-US-Great_Plains/test_data_in/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Great_Plains \
    --pred-path generated/regression/part1-US-Great_Plains/test_data_in_ERA5/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/generation-network-snapshot-000242.nc \
    --out-dir results/regression/part1-US-Great_Plains/test_data_in_ERA5/00007-Regression-MyDataset-part1-US-Great_Plains-ddpmpp-20260428_110416/

# part1-US-Nevada
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Nevada \
    --pred-path generated/regression/part1-US-Nevada/test_data_in/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/generation-network-snapshot-000242.nc \
    --out-dir results/regression/part1-US-Nevada/test_data_in/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Nevada \
    --pred-path generated/regression/part1-US-Nevada/test_data_in_ERA5/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/generation-network-snapshot-000242.nc \
    --out-dir results/regression/part1-US-Nevada/test_data_in_ERA5/00009-Regression-MyDataset-part1-US-Nevada-ddpmpp-20260428_195807/

# part1-US-Rocky
python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Rocky \
    --pred-path generated/regression/part1-US-Rocky/test_data_in/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/generation-network-snapshot-000242.nc \
    --out-dir results/regression/part1-US-Rocky/test_data_in/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part1-US-Rocky \
    --pred-path generated/regression/part1-US-Rocky/test_data_in_ERA5/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/generation-network-snapshot-000242.nc \
    --out-dir results/regression/part1-US-Rocky/test_data_in_ERA5/00005-Regression-MyDataset-part1-US-Rocky-ddpmpp-20260428_002708/

# part2-US-Appalachian
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/regression/part2-US-Appalachian/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Appalachian/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/regression/part2-US-Appalachian/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Appalachian/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/regression/part2-US-Appalachian/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Appalachian/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/regression/part2-US-Appalachian/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Appalachian/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Appalachian \
    --pred-path generated/regression/part2-US-Appalachian/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Appalachian/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/

# part2-US-Cascades
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/regression/part2-US-Cascades/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Cascades/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/regression/part2-US-Cascades/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Cascades/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/regression/part2-US-Cascades/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Cascades/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/regression/part2-US-Cascades/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Cascades/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Cascades \
    --pred-path generated/regression/part2-US-Cascades/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Cascades/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/

# part2-US-Great_Plains
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/regression/part2-US-Great_Plains/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Great_Plains/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/regression/part2-US-Great_Plains/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Great_Plains/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/regression/part2-US-Great_Plains/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Great_Plains/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/regression/part2-US-Great_Plains/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Great_Plains/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Great_Plains \
    --pred-path generated/regression/part2-US-Great_Plains/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Great_Plains/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/

# part2-US-Nevada
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/regression/part2-US-Nevada/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Nevada/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/regression/part2-US-Nevada/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Nevada/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/regression/part2-US-Nevada/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Nevada/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/regression/part2-US-Nevada/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Nevada/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Nevada \
    --pred-path generated/regression/part2-US-Nevada/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Nevada/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/

# part2-US-Rocky
python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/regression/part2-US-Rocky/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Rocky/test_data_in/00012-Regression-MyDataset-part2-US-Great_Plains-ddpmpp-20260503_000041/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/regression/part2-US-Rocky/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Rocky/test_data_in/00013-Regression-MyDataset-part2-US-Nevada-ddpmpp-20260503_000140/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/regression/part2-US-Rocky/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Rocky/test_data_in/00014-Regression-MyDataset-part2-US-Rocky-ddpmpp-20260503_000147/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/regression/part2-US-Rocky/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Rocky/test_data_in/00015-Regression-MyDataset-part2-US-Appalachian-ddpmpp-20260503_224800/

python -m metrics.run_metrics \
    --domain-dir dataset_root/part2-US-Rocky \
    --pred-path generated/regression/part2-US-Rocky/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/generation-network-snapshot-000465.nc \
    --out-dir results/regression/part2-US-Rocky/test_data_in/00016-Regression-MyDataset-part2-US-Cascades-ddpmpp-20260503_224820/