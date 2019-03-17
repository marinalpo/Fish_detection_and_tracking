from moviepy.editor import *

video_name='Andratx9_6L'
start_time=0 # seconds
end_time=30 # seconds

myvideo = VideoFileClip(video_name+'.MP4')

myvideoedited = myvideo.subclip(start_time, end_time)

myvideoedited.write_videofile(video_name+'_short.mp4',codec='libx264')
