import cv2
import json
import os

import torch
from PIL import Image
from scenedetect import detect, AdaptiveDetector
from lavis.models import load_model_and_preprocess

video_directory = '/media/dhiman/dp-hd/Education/Videos/processed_videos'
pretrain_jsonl_path = 'data/pretrain/pretrain_scene_blip.jsonl'


def load_blip_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_extract, vis_processors_extract, txt_processors_extract = load_model_and_preprocess(name="blip2_feature_extractor",
    #                                                                   model_type="pretrain", is_eval=True,
    #                                                                   device=device)  # Blip2 featrures

    # model_sim, vis_processors_sim, txt_processors_sim = load_model_and_preprocess("blip_image_text_matching", "base",
    #                                                                               device=device, is_eval=True)
    # model_extract, vis_processors_extract, txt_processors_extract = load_model_and_preprocess("blip_image_text_matching", "large", device=device, is_eval=True)

    # we associate a model with its preprocessors to make it easier for inference.
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="large_coco", is_eval=True, device=device
    )
    # uncomment to use base model
    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip_caption", model_type="base_coco", is_eval=True, device=device
    # )
    # vis_processors.keys()

    # we associate a model with its preprocessors to make it easier for inference.
    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
    # )

    # Other available models:
    #
    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
    # )
    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
    # )
    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
    # )
    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
    # )
    #
    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
    # )
    #
    # model, vis_processors, _ = load_model_and_preprocess(
    #     name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device
    # )
    return model, vis_processors, device


def read_file_names_from_directory(directory_path):
    return [f for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))]


def get_pending_files_to_process(pretrain_file_path, video_list):
    videos_to_exclude = set()

    try:
        with open(pretrain_file_path, "r") as f:
            existing_data = [json.loads(l.strip("\n")) for l in f.readlines()]
    except:
        existing_data = []

    for val in existing_data:
        videos_to_exclude.add(val["vid"])

    return [f for f in video_list
            if get_name_without_ext(f) not in videos_to_exclude]


def get_name_without_ext(file_name):
    return os.path.splitext(file_name)[0]


def read_specific_frame(video_path, frame_number):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Set the frame position to the desired frame number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame, mode="RGB")
    # image = Image.new("BGR", (frame.shape[0], frame.shape[1]))
    # color = [tuple(pixel) for row in frame for pixel in row]
    # image.putdata(frame)
    image.show()
    return image

def process_video(video_file):
    file_path = os.path.join(video_directory, video_file)
    scene_list = detect(file_path, AdaptiveDetector())
    print(scene_list)
    for scene in scene_list:
        read_specific_frame(file_path, scene[0].get_frames())


def process_videos():
    load_blip_model()
    videos = read_file_names_from_directory(video_directory)
    videos_to_process = get_pending_files_to_process(pretrain_jsonl_path, videos)
    print(videos_to_process)
    process_video(videos_to_process[0])

if __name__ == "__main__":
    process_videos()
