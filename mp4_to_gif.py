# convert mp4 to gif

import imageio
import os
import sys

def convert_mp4_to_gif(mp4_file, gif_file):
    print("Converting", mp4_file, "to", gif_file)
    reader = imageio.get_reader(mp4_file)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(gif_file, fps=fps)
    for frame in reader:
        writer.append_data(frame)
    print("Done")
    writer.close()

if __name__ == "__main__":
    lastFolder = sys.argv[1].split("/")[-2]
    for file in os.listdir(sys.argv[1]):
        if file.endswith(".mp4"):
            convert_mp4_to_gif(sys.argv[1] + file, "img/gif/" + lastFolder + ".gif")