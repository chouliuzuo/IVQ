## Data Preprocessing

Once you want to train a GVMGen model, please prepare your data following this document.

### 1. cutting

If you need to trim the data due to text overlays in the video or human voice in the background music, please refer to utils/video_cut.py. If you need to trim the music data separately, please refer to utils/music_cut.py. **Note to keep the alignment between video and music.**

### 2. video music split

```
python utils/video_music_split.py
```

Extract .wav files from original .mp4 files.

### 3. transform video into tensors

```
python videodata_pre.py
```

#### 4. Build a folder for dataset

```
python dataset_pro.py
```

After preprocessing, all videos are transformed into tensors (.pt files), music is transformed in .wav files, train/data.jsonl and eval/data.jsonl contain the index of training set and test set.
