from moviepy.editor import VideoFileClip
import os
path = "..." # a folder path
for file in os.listdir(path):
    if not file.endswith(".mp4"):
        continue
    clip = VideoFileClip(os.path.join(path, file))
    audio = clip.audio
    audio.write_audiofile(os.path.join(path, file.replace('.mp4',".wav")))
    print(file)