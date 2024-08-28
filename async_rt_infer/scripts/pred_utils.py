import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode


# Detectron2 config put in a seperate module
class detectroninference:
    def __init__(self,model_path,config_file,num_cls=1,name_classes=["ballon"]):
        self.cfg = get_cfg()
        self.cfg.set_new_allowed(True)
        assert (os.path.exists(config_file)),f"Config file {config_file} does not exist!!"
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.WEIGHTS = model_path
        self.cfg.SOLVER.IMS_PER_BATCH = 1 
        self.predictor = DefaultPredictor(self.cfg)
        self.veg_metadata = MetadataCatalog.get("ballon").set(thing_classes=name_classes)


    
    def apply_mask(self,mask,img):
        all_masks=np.zeros(mask.shape,dtype=np.uint8)
        all_patches=np.zeros((*mask.shape,3),dtype=np.uint8)
        """Apply the given mask to the image."""
        for i in range(all_masks.shape[0]):
                all_masks[i][:, :] = np.where(mask[i] == True,255,0)
                for j in range(3):
                    all_patches[i][:, :,j] = np.where(mask[i] == True,img[:,:,j],0)
        return all_masks,all_patches


    def pred(self,img):
        orig_img=img.copy()
        height,width=img.shape[:2]
        outputs = self.predictor(img)  
        print(f"Output Tensor device is {outputs['instances'].pred_masks.get_device()}")
        masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
        #masks,patches=self.apply_mask(masks,orig_img)
        classes=outputs["instances"].pred_classes.to("cpu").numpy()
        boxes=(outputs["instances"].pred_boxes.to("cpu").tensor.numpy())
        #return masks,boxes,classes
        return masks,boxes,classes,outputs["instances"].scores.to("cpu").numpy()

def rle_encode(mask):
    pixels = mask.flatten()
    # Ensure we handle the edge case where mask is empty
    if len(pixels) == 0:
        return []
    # Get the positions where the value changes
    runs = np.diff(pixels)
    run_starts = np.where(runs != 0)[0] + 1
    # Add start and end positions
    run_starts = np.concatenate(([0], run_starts, [len(pixels)]))
    run_lengths = np.diff(run_starts)
    run_values = pixels[run_starts[:-1]]
    # Combine lengths and values
    rle = np.column_stack((run_lengths, run_values)).flatten()
    return rle.tolist()
