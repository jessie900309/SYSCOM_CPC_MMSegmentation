from mmseg.apis import inference_segmentor
import mmcv
import cv2
import os


def convertMP4(inputMP4, outputDir, opacity, model):
    input_video = inputMP4
    video = mmcv.VideoReader(input_video)
    img_index = 0
    input_video_name = os.path.basename(input_video)[:-4]
    for frame in video:
        result = inference_segmentor(model, frame)
        save_img = '{}/{}/result{}.jpg'.format(outputDir, input_video_name, img_index)
        model.show_result(frame, result, out_file=save_img, opacity=opacity)
        img_index += 1


def convertJPGtoMP4(inputMP4, outputDir, model):
    input_video = inputMP4
    input_video_name = os.path.basename(input_video)[:-4]
    input_video_frame_dir = outputDir + '/' + input_video_name
    # get all image
    image_list = os.listdir(input_video_frame_dir)
    print("len of image list = ", len(image_list))
    # get image W/H
    img = cv2.imread('{}/{}'.format(input_video_frame_dir, image_list[0]))
    if img is not None:
        print('{}/{}'.format(input_video_frame_dir, image_list[0]))
        size = (img.shape[1], img.shape[0])
        print("size = ", size)
        # get FPS
        video = cv2.VideoCapture(input_video)
        fps = video.get(cv2.CAP_PROP_FPS)
        print("fps = ", fps)
        # write MP4
        output_dir = 'output_video'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videowrite = cv2.VideoWriter('{}/result{}.mp4'.format(output_dir, input_video_name), fourcc, fps, size)
        for img in image_list:
            img_path = "{}/".format(input_video_frame_dir) + img
            img = cv2.imread(img_path)
            videowrite.write(img)
        # output
        videowrite.release()
        print('output video : {}/result{}.mp4'.format(output_dir, input_video_name))
    else:
        print('img is None')
        print('{}/{}'.format(input_video_frame_dir, image_list[0]))


if __name__ == '__main__':
    convertMP4("nothingOuO")
    convertJPGtoMP4("nothingOuO")
