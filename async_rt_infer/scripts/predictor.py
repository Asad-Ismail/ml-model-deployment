from __future__ import print_function
import os
import json
from io import StringIO
import sys
import signal
import traceback
from flask import Flask, request, Response
import warnings
import numpy as np
from pred_utils import detectroninference, rle_encode

# Suppress future warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

# Model path
prefix = '/opt/ml/model/'
model_path = os.path.join(prefix, 'model_final.pth')
cfg_path = os.path.join(prefix, 'model_final.yaml')
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

# Preload model on startup
ScoringService.get_model()

# The flask app for serving predictions
app = Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy."""
    health = ScoringService.get_model() is not None
    status = 200 if health else 404
    return Response(response=f'Successfully Pinged Health status {status}\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data."""
    try:
        model = ScoringService.get_model()
        if model is None:
            return Response(response='Model could not load\n', status=500, mimetype='application/json')

        payload = json.loads(request.data)
        img = np.array(payload["images"], dtype=np.uint8)
        print("Running Inference!!")
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

        result = json.dumps(response)
        return Response(response=result, status=200, mimetype="application/json")

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        traceback.print_exc()
        return Response(response='{"error": "Error during inference"}\n', status=500, mimetype='application/json')

if __name__ == "__main__":
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()

