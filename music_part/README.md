## IVQ: Structured and Lightweight Vector Quantization via Binary Hierarchical Composition Inspired by *IChing*


### 1. Environment Preparation

Please first clone the repo and install the required environment, which can be done by running the following commands:

```
conda env create -n IVQ python=3.9.0

conda activate IVQ

cd IVQ

pip install -r requirements.txt
```

### 2. Data Preprocessing

Please refer to ```./data_preprocess```

### 3. Training

* Modify the config files
  There are some variables you **must** modify before your training. Other changes are optional and you can refer to each ```default.yaml```

  ```
  config/dset/train.yaml datasource.evaluate path/to/eval_folder
  config/dset/train.yaml datasource.generate path/to/eval_folder
  config/dset/train.yaml datasource.train path/to/train_folder
  config/dset/train.yaml datasource.valid path/to/eval_folder

  config/solver/gvmgen/gvmgen.yaml compression_model_checkpoint path/to/IVQ_compression_model
  config/teams/default.yaml default.dora_dir path/to/GVMGen
  config/teams/default.yaml default.reference_dir path/to/GVMGen
  config/teams/default.yaml darwin.dora_dir path/to/GVMGen
  config/teams/default.yaml darwin.reference_dir path/to/GVMGen
  ```
* run reconstruction training

  ```
  bash run_train.sh
  ```

* run v2m training

  ```
  bash run_v2m.sh
  ```

### 4. Inference

* transform model weights (Run this step **only when loading your own trained model**. If you want to test our published model, please skip it.)

```
python load_model.py --checkpoint_path path/to/your_checkpoint --output_path path/to/output
```
* reconstruction (Please modify variables ```name```, ```output_dir```, ```music_dir``` into checkpoint directory, output directory and original music directory separately in ```reconstruction.py```)

```
python reconstruction.py
```

* inference

```
python v2m.py --model_path ./checkpoints --video_pt_dir /path/to/video_pt --syn_path output --fps 1 --duration 30
```


### 5. Model weights

The pretrained model's parameter weights have been published and can be accessed at [here](https://huggingface.co/chouliu/IVQ/tree/main), in which ```compression_state_dict.bin``` is for music reconstruction while ```state_dict.bin``` is for video-to-music generation. You should put these two models in the same directory and set the ```model_path``` parameter to the directory for whether ```reconstruction.py``` or ```v2m.py```
