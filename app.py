from PIL.ImageOps import colorize, scale
import gradio as gr
import importlib
import sys
import os
import pdb
from matplotlib.pyplot import step
from model_args import segtracker_args,sam_args,aot_args
from SegTracker import SegTracker
from tool.transfer_tools import draw_outline, draw_points
import cv2
from PIL import Image
from skimage.morphology.binary import binary_dilation
import argparse
import torch
import time, math
from seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
import gc
import numpy as np
import json
from tool.transfer_tools import mask2bbox


def tracking_objects(Seg_Tracker, input_video, frame_num=0):
    fps = 8
    print("Start tracking !")
    # pdb.set_trace()
    # output_video, output_mask=tracking_objects_in_video(Seg_Tracker, input_video, input_img_seq, fps)
    # pdb.set_trace()
    return tracking_objects_in_video(Seg_Tracker, input_video,fps, frame_num)

def gd_detect(Seg_Tracker, origin_frame, grounding_caption):
    box_threshold = 0.5
    text_threshold = 0.5
    aot_model = "r50_deaotl"
    long_term_mem = 9999
    max_len_long_term = 9999
    sam_gap = 9999
    max_obj_num = 255
    points_per_side = 16
    if Seg_Tracker is None:
        Seg_Tracker, _ , _, _ = init_SegTracker(origin_frame)
    print("Detect")
    predicted_mask, annotated_frame= Seg_Tracker.detect_and_seg(origin_frame, grounding_caption, box_threshold, text_threshold)
    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    masked_frame = draw_mask(annotated_frame, predicted_mask)
    return Seg_Tracker, masked_frame, origin_frame

def get_meta_from_video(Seg_Tracker, input_video, grounding_caption):
    if input_video is None:
        return None, None, None, ""
    print("get meta information of input video")
    cap = cv2.VideoCapture(input_video)
    _, first_frame = cap.read()
    cap.release()
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    Seg_Tracker, masked_frame, origin_frame = gd_detect(Seg_Tracker, first_frame, grounding_caption)
    painted_video, masked_video_path, masked_images_folder_path, origional_images_folder_path, zip_path = tracking_objects(Seg_Tracker, input_video, frame_num=0)
    print(masked_images_folder_path)
    print(origional_images_folder_path)
    os.system("python ProPainter/inference_propainter.py  --video "+ origional_images_folder_path + " --mask " + masked_images_folder_path)
    #os.system("python /content/Segment-and-Track-Anything/ProPainter/inference_propainter.py --video /content/Segment-and-Track-Anything/tracking_results/blackswan/blackswan_masked_frames --mask /content/Segment-and-Track-Anything/tracking_results/blackswan/blackswan_masks")
    return masked_video_path, painted_video

def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
    with torch.cuda.amp.autocast():
        # Reset the first frame's mask
        frame_idx = 0
        Seg_Tracker.restart_tracker()
        Seg_Tracker.add_reference(origin_frame, predicted_mask, frame_idx)
        Seg_Tracker.first_frame_mask = predicted_mask
    return Seg_Tracker
def init_SegTracker(origin_frame):
    aot_model = "r50_deaotl"
    long_term_mem = 9999
    max_len_long_term = 9999
    sam_gap = 9999
    max_obj_num = 255
    points_per_side = 16
    if origin_frame is None:
        return None, origin_frame, [[], []], ""
    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term
    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side

    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()

    return Seg_Tracker, origin_frame, [[], []], ""

def init_SegTracker_Stroke(aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side, origin_frame):
    if origin_frame is None:
        return None, origin_frame, [[], []], origin_frame
    # reset aot args
    aot_args["model"] = aot_model
    aot_args["model_path"] = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"] = long_term_mem
    aot_args["max_len_long_term"] = max_len_long_term
    # reset sam args
    segtracker_args["sam_gap"] = sam_gap
    segtracker_args["max_obj_num"] = max_obj_num
    sam_args["generator_args"]["points_per_side"] = points_per_side
    Seg_Tracker = SegTracker(segtracker_args, sam_args, aot_args)
    Seg_Tracker.restart_tracker()
    return Seg_Tracker, origin_frame, [[], []], origin_frame

def undo_click_stack_and_refine_seg(Seg_Tracker, origin_frame, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side):
    if Seg_Tracker is None:
        return Seg_Tracker, origin_frame, [[], []]
    print("Undo!")
    if len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][: -1]
        click_stack[1] = click_stack[1][: -1]
    if len(click_stack[0]) > 0:
        prompt = {
            "points_coord":click_stack[0],
            "points_mode":click_stack[1],
            "multimask":"True",
            }
        masked_frame = seg_acc_click(Seg_Tracker, prompt, origin_frame)
        return Seg_Tracker, masked_frame, click_stack
    else:
        return Seg_Tracker, origin_frame, [[], []]

def roll_back_undo_click_stack_and_refine_seg(Seg_Tracker, origin_frame, click_stack, aot_model, long_term_mem, max_len_long_term, sam_gap, max_obj_num, points_per_side,input_video, input_img_seq, frame_num, refine_idx):
    if Seg_Tracker is None:
        return Seg_Tracker, origin_frame, [[], []]
    print("Undo!")
    if len(click_stack[0]) > 0:
        click_stack[0] = click_stack[0][: -1]
        click_stack[1] = click_stack[1][: -1]
    if len(click_stack[0]) > 0:
        prompt = {
            "points_coord":click_stack[0],
            "points_mode":click_stack[1],
            "multimask":"True",
        }
        chosen_frame_show, curr_mask, ori_frame = res_by_num(input_video, input_img_seq, frame_num)
        Seg_Tracker.curr_idx = refine_idx
        predicted_mask, masked_frame = Seg_Tracker.seg_acc_click(
                                                        origin_frame=origin_frame,
                                                        coords=np.array(prompt["points_coord"]),
                                                        modes=np.array(prompt["points_mode"]),
                                                        multimask=prompt["multimask"],
                                                        )
        curr_mask[curr_mask == refine_idx]  = 0
        curr_mask[predicted_mask != 0]  = refine_idx
        predicted_mask=curr_mask
        Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
        return Seg_Tracker, masked_frame, click_stack
    else:
        return Seg_Tracker, origin_frame, [[], []]


def seg_acc_click(Seg_Tracker, prompt, origin_frame):
    # seg acc to click
    predicted_mask, masked_frame = Seg_Tracker.seg_acc_click(
                                                      origin_frame=origin_frame,
                                                      coords=np.array(prompt["points_coord"]),
                                                      modes=np.array(prompt["points_mode"]),
                                                      multimask=prompt["multimask"],
                                                    )

    Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
    return masked_frame

def add_new_object(Seg_Tracker):

    prev_mask = Seg_Tracker.first_frame_mask
    Seg_Tracker.update_origin_merged_mask(prev_mask)
    Seg_Tracker.curr_idx += 1

    print("Ready to add new object!")

    return Seg_Tracker, [[], []]



def seg_track_app():
    ##########################################################
    ######################  Front-end ########################
    ##########################################################
    app = gr.Blocks(theme = gr.themes.Monochrome())
    with app:
        gr.Markdown(
            '''
            <div style="text-align:center;">
                <span style="font-size:3em; font-weight:bold;">Pro Painter Tool</span>
            </div>
            '''
        )
        click_stack = gr.State([[],[]]) # Storage clicks status
        origin_frame = gr.State(None)
        Seg_Tracker = gr.State(None)
        current_frame_num = gr.State(None)
        refine_idx = gr.State(None)
        frame_num = gr.State(None)
        aot_model = gr.State(None)
        sam_gap = gr.State(None)
        points_per_side = gr.State(None)
        max_obj_num = gr.State(None)
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                   gr.Markdown("## Pro Painter By EKKel AI")
                with gr.Row():
                    input_video = gr.Video(label="Origional Video").style(height="200px")
                    masked_video = gr.Video(label="Masked Video").style(height="200px")
                    output_video = gr.Video(label="Output Video").style(height="200px")
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    grounding_caption = gr.Textbox(label="Detection Prompt")
        with gr.Row(variant="panel"):
            with gr.Column():
                with gr.Row():
                    detect_button = gr.Button(value = "Paint Object")
                    new_object_button = gr.Button(value="Add new object", interactive=True)

    ##########################################################
    ######################  back-end #########################
    ##########################################################
        #-------------- Input compont -------------
        # Use grounding-dino to detect object
        detect_button.click(
            fn=get_meta_from_video,
            inputs=[
                Seg_Tracker, input_video, grounding_caption
                ],
            outputs = [masked_video, output_video]
                )
        # Add new object
        new_object_button.click(
            fn=add_new_object,
            inputs=
            [
                Seg_Tracker
            ],
            outputs=
            [
                Seg_Tracker, click_stack
            ]
        )
        with gr.Tab(label='Video example'):
            gr.Examples(
                examples=[
                    os.path.join(os.path.dirname(__file__), "assets", "blackswan.mp4"),
                    os.path.join(os.path.dirname(__file__), "assets", "cars.mp4"),
                    os.path.join(os.path.dirname(__file__), "assets", "cell.mp4"),
                    ],
                inputs=[input_video],
            )
    app.queue(concurrency_count=1)
    app.launch(debug=True, enable_queue=True, share=True)
if __name__ == "__main__":
    seg_track_app()