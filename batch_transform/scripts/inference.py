from __future__ import print_function
import os
import json
import traceback
import numpy as np
from pred_utils import detectroninference, rle_encode

# Model path
prefix = '/opt/ml/model/'
model_path = os.path.join(prefix, 'model_final.pth')
cfg_path = os.path.join(prefix, 'pred_config.yaml')
print(f"The Model Weights path exists? {os.path.exists(model_path)}")
print(f"The Config path exists? {os.path.exists(cfg_path)}")

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            print("Initializing the model for Inference!!")
            cls.model = detectroninference(model_path, cfg_path, name_classes=["BP", "HP"])
        return cls.model

# Initialize model at the start
ScoringService.get_model()

def transform_fn(input_data, content_type, accept_type):
    """
    Transform function to handle prediction requests.
    """
    try:
        # Parse input data
        payload = json.loads(input_data)
        img = np.array(payload["images"], dtype=np.uint8)

        # Run inference
        print("Running Inference!!")
        model = ScoringService.get_model()
        masks, boxes, classes, scores = model.pred(img)

        if len(masks) == 0:
            response = {"masks": [], "boxes": [], "classes": [], "scores": [], "mask_shape": None}
        else:
            rle_masks = [rle_encode(mask) for mask in masks]
            response = {
                "masks": rle_masks,
                "boxes": boxes.tolist(),
                "classes": classes.tolist(),
                "scores": scores.tolist(),
                "mask_shape": masks[0].shape
            }

        # Serialize response
        return json.dumps(response), accept_type

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        traceback.print_exc()
        return json.dumps({"error": "Error during inference"}), 'application/json'

def handle_batch_request(input_dir, output_dir):
    """
    Handles batch processing of files for AWS Batch Transform.
    """
    for input_file in os.listdir(input_dir):
        if input_file.endswith('.json'):
            input_path = os.path.join(input_dir, input_file)
            output_path = os.path.join(output_dir, input_file.replace('.json', '.out'))

            with open(input_path, 'r') as f:
                input_data = f.read()

            # Process the input file
            output_data, _ = transform_fn(input_data, 'application/json', 'application/json')

            # Write the output to the output directory
            with open(output_path, 'w') as out_f:
                out_f.write(output_data)

if __name__ == "__main__":
    input_dir = os.environ.get('SM_INPUT_DIR', '/opt/ml/input/data/')
    output_dir = os.environ.get('SM_OUTPUT_DIR', '/opt/ml/output/data/')
    handle_batch_request(input_dir, output_dir)

