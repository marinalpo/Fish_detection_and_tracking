from moviepy.editor import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('original_video_path', help = "Path of the original video (without final slash)")
parser.add_argument('final_video_path', help = "Path of the destination video (without final slash)")
parser.add_argument('video_name', help = "Name of the video (without .MP4)")
parser.add_argument('initial_second', type=int, help = "Second in the original video where the cut will begin")
parser.add_argument('final_second', type=int, help = "Second in the original video where the cut will end")
args = parser.parse_args()

myvideo = VideoFileClip(args.original_video_path+'/'+args.video_name+'.MP4')

myvideoedited = myvideo.subclip(args.initial_second, args.final_second)

myvideoedited.write_videofile(args.final_video_path+'/'+args.video_name+'_short.mp4', codec='libx264')