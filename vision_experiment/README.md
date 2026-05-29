# Run scripts

## Stage1
```sh
WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=4 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_titok.py config=configs/training/TiTok/stage1/titok_b64.yaml \
    experiment.project="titok_b64_stage1" \
    experiment.name="titok_b64_stage1_run1" \
    experiment.output_dir="titok_b64_stage1_run1" \
    training.per_gpu_batch_size=16
```

## Stage2

```sh
WANDB_MODE=offline accelerate launch --num_machines=1 --num_processes=4 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_titok.py config=configs/training/TiTok/stage2/titok_b64.yaml \
    experiment.project="titok_b64_stage2" \
    experiment.name="titok_b64_stage2_run1" \
    experiment.output_dir="titok_b64_stage2_run1" \
    training.per_gpu_batch_size=16 \
    experiment.init_weight="/path/to/stage1_pytorch_model.bin"
```

# Score Scripts

## Reconstruct
```sh
python score/reconstruct.py \
    --input_dir /path/to/input_images \
    --output_dir ./outputs \
    --weight /path/to/model
```

## Score

```sh
python score/score.py \
    --original_dir /path/to/input_images \
    --reconstruct_dir ./outputs
```