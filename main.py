import os
from convert_function import convertMP4, convertJPGtoMP4
from load_model import load_MMSmodel


input_video_dir = 'syscom_video/B8DF6B001667'
mms_opacity = 0.5 # 0~1


def main():
    # load model
    MMSmodel = load_MMSmodel()
    # frame folder
    output_frame_dir = 'syscom_video_frame/' + os.path.basename(input_video_dir)
    if not os.path.exists(output_frame_dir):
        os.makedirs(output_frame_dir)
    # result
    for input_video in os.listdir(input_video_dir):
        input_video_path = input_video_dir + '/' + input_video
        print("now convert " + input_video_dir + '/' + input_video + '...')
        convertMP4(input_video_path, output_frame_dir, mms_opacity, MMSmodel)
        print("convert image list to video...")
        convertJPGtoMP4(input_video_path, output_frame_dir)


if __name__ == '__main__':
    main()
