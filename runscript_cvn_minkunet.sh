# # from the repo root
# export PYTHONPATH=.

# pick two GPUs
export CUDA_VISIBLE_DEVICES=1
  # train.dataset_path="Custom:root=/nfs/data/1/nitish/imagenetood" \
  # crops.global_crops_size=168 crops.local_crops_size=56 crops.local_crops_number=8 student.patch_size=14 \
  # MODEL.WEIGHTS=/nfs/data/1/nitish/dino_output/cvn_properrun/model_0009999.rank_0.pth \

# accum_iter isn't actually doing anything right now, TODO : get it working
torchrun --standalone --nproc_per_node=1 \
  dinov2/train/train.py \
  --config-file dinov2/configs/ssl_my_config.yaml \
  --output-dir /nfs/data/1/nitish/dino_output/cvn_minkunet_trywarp2 \
  train.dataset_path="CVN:root=/nfs/data/1/rrazakami/work/data_cvn/data/dune/2023_trainings/latest/dunevd:extra=plane=Z,mono=mono1" \
  train.num_workers=2 \
  crops.global_crops_size=200 crops.local_crops_size=96 crops.local_crops_number=0 student.patch_factor=1 \
  train.batch_size_per_gpu=64 train.accum_iter=4 \
  model.mixed_precision.param_dtype=fp32 model.mixed_precision.reduce_dtype=fp32 model.mixed_precision.buffer_dtype=fp32 \
  train.autocast_dtype=fp32 \
  fsdp.use_fsdp=false model.use_fsdp=false train.use_fsdp=false \
  dino.koleo_loss_weight=0.0 optim.base_lr=0.0008 optim.clip_grad=2.0 \
  optim.adamw_beta2=0.99 student.drop_path_rate=0.1 student.arch=minkunet \
  ibot.separate_head=true ibot.head_n_prototypes=64 ibot.head_nlayers=1 \
  dino.head_n_prototypes=64 dino.head_nlayers=1 \
  student.patch_size=1
