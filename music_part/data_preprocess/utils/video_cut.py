import os
import wave
from moviepy.editor import VideoFileClip


input_path = '...'
output_path = '...'
video = VideoFileClip(input_path)
start_time = 0
end_time = 1
segment_video = video.subclip(start_time, end_time)
segment_video.write_videofile(output_path, codec='libx264')
