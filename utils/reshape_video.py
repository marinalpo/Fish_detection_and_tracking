import moviepy.editor as mp

video_name='Andratx9_6L_short'

clip = mp.VideoFileClip(video_name+'.mp4')

clip_resized = clip.resize(height=360) # the width is automatically so that the ratio is preserved

clip_resized.write_videofile(video_name+'_resized.mp4')