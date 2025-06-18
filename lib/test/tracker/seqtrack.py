from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.seqtrack_utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.models.seqtrack import build_seqtrack
from lib.test.tracker.seqtrack_utils import Preprocessor
from lib.utils.box_ops import clip_box
import numpy as np


class SEQTRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SEQTRACK, self).__init__(params)
        network = build_seqtrack(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.seq_format = self.cfg.DATA.SEQ_FORMAT
        self.num_template = self.cfg.TEST.NUM_TEMPLATES
        self.bins = self.cfg.MODEL.BINS
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.hanning = torch.tensor(np.hanning(self.bins)).unsqueeze(0).cuda()
            self.hanning = self.hanning
        else:
            self.hanning = None
        self.start = self.bins + 1 # start token
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.debug = params.debug
        self.frame_id = 0

        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)
        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)



    def initialize(self, image, info: dict):
        # get the initial templates

        z_patch_arr, _ = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)

        template = self.preprocessor.process(z_patch_arr)
        self.template_list = [template] * self.num_template

        # get the initial sequence i.e., [start]
        batch = template.shape[0]
        self.init_seq = (torch.ones([batch, 1]).to(template) * self.start).type(dtype=torch.int64)
        # What to Expect:
        #
        #     z_patch_arr: Cropped template image (e.g., 128x128x3)
        #     template: Preprocessed tensor (1x3x128x128)
        #     self.init_seq: Start token tensor [[4001]] (bins+1)
        #     self.template_list: List containing initial template
        #
        # Verification:
        #
        #     Check template dimensions match config
        #     Verify start token value equals self.bins + 1
        #     Confirm self.state contains initial bbox [x,y,w,h]

        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        images_list = self.template_list + [search]
        # What to Expect:
        #
        #     x_patch_arr: Search region crop (e.g., 256x256x3)
        #     resize_factor: Scale factor for coordinate mapping (e.g., 2.0)
        #     search: Preprocessed search tensor (1x3x256x256)
        #     images_list: Combined templates + search (e.g., length 2)
        #
        # Verification:
        #
        #     Search region centered on previous prediction
        #     Resize factor > 1 indicates upsampling
        #     Image list contains all templates plus current search


        with torch.no_grad():
            xz = self.network.forward_encoder(images_list)
        # run the encoder
        # What to Expect:
        #
        #     xz: List of encoded features from different layers
        #     xz[-1]: Final layer features (1x(N_template + N_search)xD)
        #     Feature dimensions: e.g., (1, 320, 768) for templates(64) + search(256) patches
        #
        # Verification:
        #
        #     Check feature dimensions match expected patch counts
        #     Verify no gradient computation (inference mode)
        #     Ensure all templates and search are processed jointly
        with torch.no_grad():
            out_dict = self.network.inference_decoder(xz=xz,
                                                      sequence=self.init_seq,
                                                      window=self.hanning,
                                                      seq_format=self.seq_format)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        # run the decoder
        # What to Expect:
        #
        #     out_dict['pred_boxes']: Generated coordinates (1x4) in [0, bins-1] range
        #     out_dict['confidence']: Sequence generation confidence scores
        #     pred_boxes: Normalized predictions ready for mapping
        #
        # Verification:
        #
        #     Coordinates should be in valid range [0, bins-1]
        #     Confidence scores indicate generation quality
        #     Shape matches expected (batch_size, 4)

        # if use other formats of sequence
        if self.seq_format == 'corner':
            pred_boxes = box_xyxy_to_cxcywh(pred_boxes)
        if self.seq_format == 'whxy':
            pred_boxes = pred_boxes[:, [2, 3, 0, 1]]

        pred_boxes = pred_boxes / (self.bins-1)
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]

        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # What to Expect:
        #
        #     pred_boxes: Normalized to [0,1] range
        #     pred_box: Scaled to search region coordinates
        #     self.state: Final bbox in image coordinates [x,y,w,h]
        #
        # Verification:
        #
        #     Coordinates properly mapped from bins to pixels
        #     Bounding box within image boundaries
        #     State update reflects current frame prediction

        # update the template
        if self.num_template > 1:
            conf_score = out_dict['confidence'].sum().item() * 10 # the confidence score
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, _ = sample_target(image, self.state, self.params.template_factor,
                                               output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)
            # What to Expect:
            #
            #     conf_score: Tracking confidence (higher = better)
            #     Template update triggered by interval + confidence threshold
            #     self.template_list: Updated with new template, maintaining size limit
            #
            # Verification:
            #
            #     Update only occurs when both conditions met
            #     First template (index 0) is never removed
            #     Template list size doesn't exceed self.num_template

        # for debug
        if self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return SEQTRACK
