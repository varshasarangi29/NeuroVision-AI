import os
import tempfile
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from flask import request, jsonify

def register_upload_api(app, model, DEVICE):

    @app.route("/api/upload-analyze", methods=["POST"])
    def upload_analyze():

        mods = ["t1n", "t1c", "t2w", "t2f"]
        missing = [m for m in mods if m not in request.files]

        if missing:
            return jsonify({
                "success": False,
                "error": "Missing: " + ", ".join(missing)
            }), 400

        try:
            volumes = []
            shapes = {}

            with tempfile.TemporaryDirectory() as tmp:

                for mod in mods:
                    file = request.files[mod]
                    name = file.filename.lower()

                    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
                        raise ValueError(mod + " must be .nii or .nii.gz")

                    path = os.path.join(tmp, file.filename)
                    file.save(path)

                    vol = nib.load(path).get_fdata().astype(np.float32)

                    if vol.ndim != 3:
                        raise ValueError(mod + " must be a 3D MRI volume")

                    vol = np.nan_to_num(vol)
                    shapes[mod] = list(vol.shape)

                    mask = vol != 0

                    if mask.any():
                        mean = vol[mask].mean()
                        std = vol[mask].std()

                        if std > 0:
                            vol[mask] = (vol[mask] - mean) / std

                    volumes.append(vol)

            if len(set(tuple(v.shape) for v in volumes)) != 1:
                raise ValueError("All four MRI modalities must have matching shapes")

            x = np.stack(volumes, axis=0)

            x = torch.from_numpy(x).unsqueeze(0).float()

            x = F.interpolate(
                x,
                size=(64, 64, 64),
                mode="trilinear",
                align_corners=False
            ).to(DEVICE)

            model.eval()

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            classes = ["Low", "Moderate", "High"]
            pred_idx = int(np.argmax(probs))

            probabilities = {
                classes[i]: round(float(probs[i]) * 100, 2)
                for i in range(3)
            }

            return jsonify({
                "success": True,
                "prediction": classes[pred_idx],
                "confidence": probabilities[classes[pred_idx]],
                "probabilities": probabilities,
                "original_shapes": shapes,
                "model_input_shape": [4, 64, 64, 64],
                "device": str(DEVICE),
                "disclaimer":
                    "Dataset-relative tumor burden category; not a clinical tumor grade or diagnosis."
            })

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400
