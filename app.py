
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from PIL import Image

from flask import Flask, jsonify, request, render_template


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best_pediatric_3d_resnet.pth"
)

PREDICTION_CSV = os.path.join(
    BASE_DIR,
    "final_test_predictions.csv"
)

UNCERTAINTY_CSV = os.path.join(
    BASE_DIR,
    "boundary_uncertainty_results.csv"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESSED_DIR = os.path.join(
    BASE_DIR,
    "preprocessed_3d"
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class BasicBlock3D(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1
    ):

        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm3d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm3d(
            out_channels
        )

        self.downsample = None

        if (
            stride != 1
            or
            in_channels != out_channels
        ):

            self.downsample = nn.Sequential(

                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm3d(
                    out_channels
                )
            )


    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):

    def __init__(
        self,
        block=BasicBlock3D,
        layers=(2, 2, 2, 2),
        num_classes=3,
        in_channels=4
    ):

        super().__init__()

        self.in_channels = 16

        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                16,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.layer1 = self._make_layer(
            block,
            16,
            layers[0],
            stride=1
        )

        self.layer2 = self._make_layer(
            block,
            32,
            layers[1],
            stride=2
        )

        self.layer3 = self._make_layer(
            block,
            64,
            layers[2],
            stride=2
        )

        self.layer4 = self._make_layer(
            block,
            128,
            layers[3],
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d(
            (1, 1, 1)
        )

        self.dropout = nn.Dropout(
            0.30
        )

        self.fc = nn.Linear(
            128,
            num_classes
        )


    def _make_layer(
        self,
        block,
        out_channels,
        blocks,
        stride
    ):

        layers = []

        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride
            )
        )

        self.in_channels = out_channels

        for _ in range(
            1,
            blocks
        ):

            layers.append(
                block(
                    self.in_channels,
                    out_channels
                )
            )

        return nn.Sequential(
            *layers
        )


    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.dropout(x)

        x = self.fc(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cpu")

model = ResNet3D(
    num_classes=3,
    in_channels=4
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.to(device)
model.eval()


LABEL_NAMES = {
    0: "Low",
    1: "Moderate",
    2: "High"
}


# ============================================================
# LOAD RESULT TABLES
# ============================================================

prediction_df = pd.read_csv(
    PREDICTION_CSV
)

uncertainty_df = pd.read_csv(
    UNCERTAINTY_CSV
)


# ============================================================
# REAL MODEL INFERENCE
# ============================================================

def run_inference(patient_id):

    volume_path = os.path.join(
        PREPROCESSED_DIR,
        f"patient_{patient_id}.pt"
    )

    if not os.path.exists(
        volume_path
    ):

        raise FileNotFoundError(
            f"Patient {patient_id} volume not found."
        )


    volume = torch.load(
        volume_path,
        map_location="cpu"
    ).float()


    if tuple(volume.shape) != (
        4,
        64,
        64,
        64
    ):

        raise ValueError(
            f"Unexpected volume shape: {volume.shape}"
        )


    x = (
        volume
        .unsqueeze(0)
        .to(device)
    )


    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0].cpu().numpy()


    predicted_label = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[
            predicted_label
        ]
    )


    return {

        "patient_id":
            int(patient_id),

        "predicted_label":
            predicted_label,

        "predicted_class":
            LABEL_NAMES[
                predicted_label
            ],

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "probabilities": {

            "Low":
                round(
                    float(
                        probabilities[0]
                    ) * 100,
                    2
                ),

            "Moderate":
                round(
                    float(
                        probabilities[1]
                    ) * 100,
                    2
                ),

            "High":
                round(
                    float(
                        probabilities[2]
                    ) * 100,
                    2
                )
        }
    }


# ============================================================
# ADD RESEARCH INFORMATION
# ============================================================

def enrich_result(result):

    pid = result[
        "patient_id"
    ]


    # --------------------------------------------
    # Ground truth / burden
    # --------------------------------------------

    rows = prediction_df[
        prediction_df[
            "patient_id"
        ] == pid
    ]


    if len(rows) > 0:

        row = rows.iloc[0]

        result[
            "true_class"
        ] = str(
            row[
                "true_class"
            ]
        )

        result[
            "tumor_burden_percent"
        ] = round(
            float(
                row[
                    "tumor_burden_percent"
                ]
            ),
            4
        )

        result[
            "correct_prediction"
        ] = bool(
            row[
                "true_label"
            ]
            ==
            result[
                "predicted_label"
            ]
        )


    # --------------------------------------------
    # Boundary uncertainty
    # --------------------------------------------

    uncertainty_rows = (
        uncertainty_df[
            uncertainty_df[
                "patient_id"
            ] == pid
        ]
    )


    if len(
        uncertainty_rows
    ) > 0:

        u = uncertainty_rows.iloc[0]

        result[
            "boundary_distance"
        ] = round(
            float(
                u[
                    "boundary_distance"
                ]
            ),
            4
        )

        result[
            "uncertainty_score"
        ] = round(
            float(
                u[
                    "combined_uncertainty_score"
                ]
            ),
            4
        )

        result[
            "reliability_status"
        ] = str(
            u[
                "reliability_status"
            ]
        )


    return result


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)


@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# AVAILABLE TEST PATIENTS
# ============================================================

@app.route(
    "/api/patients"
)
def patients():

    available = []

    for _, row in (
        prediction_df.iterrows()
    ):

        pid = int(
            row["patient_id"]
        )

        path = os.path.join(
            PREPROCESSED_DIR,
            f"patient_{pid}.pt"
        )

        if os.path.exists(
            path
        ):

            available.append({

                "patient_id":
                    pid,

                "true_class":
                    row[
                        "true_class"
                    ],

                "burden":
                    round(
                        float(
                            row[
                                "tumor_burden_percent"
                            ]
                        ),
                        4
                    )
            })


    return jsonify(
        available
    )


# ============================================================
# ANALYZE PATIENT
# ============================================================

@app.route(
    "/api/analyze/<int:patient_id>",
    methods=["GET"]
)
def analyze(
    patient_id
):

    try:

        result = run_inference(
            patient_id
        )

        result = enrich_result(
            result
        )

        result[
            "disclaimer"
        ] = (
            "Research and educational prototype. "
            "Not for clinical diagnosis."
        )

        return jsonify(
            result
        )


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500



# ============================================================
# MRI VIEWER
# ============================================================

MODALITY_NAMES = [
    "T1N",
    "T1C",
    "T2W",
    "T2F"
]


def slice_to_base64(slice_array):

    arr = np.asarray(
        slice_array,
        dtype=np.float32
    )

    finite = np.isfinite(arr)

    if not finite.any():
        arr = np.zeros_like(
            arr,
            dtype=np.uint8
        )

    else:

        values = arr[finite]

        # Robust display scaling
        low = np.percentile(
            values,
            1
        )

        high = np.percentile(
            values,
            99
        )

        if high <= low:
            high = low + 1e-6

        arr = np.clip(
            arr,
            low,
            high
        )

        arr = (
            (arr - low)
            /
            (high - low)
            * 255.0
        )

        arr = np.nan_to_num(
            arr,
            nan=0.0,
            posinf=255.0,
            neginf=0.0
        )

        arr = arr.astype(
            np.uint8
        )


    image = Image.fromarray(
        arr,
        mode="L"
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded
    )


@app.route(
    "/api/mri/<int:patient_id>/<int:slice_index>"
)
def get_mri_slice(
    patient_id,
    slice_index
):

    try:

        path = os.path.join(
            PREPROCESSED_DIR,
            f"patient_{patient_id}.pt"
        )

        if not os.path.exists(path):

            return jsonify({
                "error":
                    "Patient volume not found"
            }), 404


        volume = torch.load(
            path,
            map_location="cpu"
        ).float()


        if tuple(volume.shape) != (
            4,
            64,
            64,
            64
        ):

            return jsonify({
                "error":
                    f"Unexpected volume shape: {tuple(volume.shape)}"
            }), 500


        # Volume shape:
        # [channel, depth, height, width]

        slice_index = max(
            0,
            min(
                int(slice_index),
                63
            )
        )


        images = {}

        for channel, name in enumerate(
            MODALITY_NAMES
        ):

            slice_array = (
                volume[
                    channel,
                    slice_index
                ]
                .cpu()
                .numpy()
            )

            images[name] = (
                slice_to_base64(
                    slice_array
                )
            )


        return jsonify({

            "patient_id":
                int(patient_id),

            "slice":
                int(slice_index),

            "total_slices":
                64,

            "volume_shape":
                [4, 64, 64, 64],

            "modalities":
                images
        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500



# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health"
)
def health():

    return jsonify({

        "status":
            "healthy",

        "model_loaded":
            True,

        "device":
            str(device),

        "test_patients":
            int(
                len(
                    prediction_df
                )
            )
    })


# ============================================================
# START SERVER
# ============================================================



# ============================================================
# PERMANENT PATIENT 172 WEB DEMO
# ============================================================

@app.route("/api/demo/patients")
def demo_patients():
    try:
        df = pd.read_csv(PREDICTION_CSV)

        patients = []
        for _, row in df.iterrows():
            patients.append({
                "patient_id": int(row["patient_id"]),
                "true_class": str(row["true_class"]),
                "predicted_class": str(row["predicted_class"])
            })

        return jsonify(patients)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/demo/analyze/<int:patient_id>")
def demo_analyze_patient(patient_id):
    try:
        df = pd.read_csv(PREDICTION_CSV)

        row = df[df["patient_id"] == patient_id]

        if row.empty:
            return jsonify({"error": "Patient not found"}), 404

        row = row.iloc[0]
        confidence = float(row["confidence_percent"])

        result = {
            "patient_id": int(row["patient_id"]),
            "true_class": str(row["true_class"]),
            "predicted_class": str(row["predicted_class"]),
            "confidence": round(confidence, 2),

            "probabilities": {
                "Low": round(float(row["prob_low"]) * 100, 2),
                "Moderate": round(float(row["prob_moderate"]) * 100, 2),
                "High": round(float(row["prob_high"]) * 100, 2)
            },

            "tumor_burden": round(
                float(row["tumor_burden_percent"]), 4
            ),

            "uncertainty": round(100 - confidence, 2),

            "reliability": (
                "High Reliability" if confidence >= 70
                else "Moderate Reliability" if confidence >= 50
                else "Low Reliability"
            )
        }

        if patient_id == 172:
            result["mri"] = {
                "t1n": "/static/mri/patient_172/t1n.png",
                "t1c": "/static/mri/patient_172/t1c.png",
                "t2w": "/static/mri/patient_172/t2w.png",
                "t2f": "/static/mri/patient_172/t2f.png"
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/demo/mri/172/<modality>")
def demo_mri_172(modality):

    allowed = ["t1n", "t1c", "t2w", "t2f"]

    modality = modality.lower()

    if modality not in allowed:
        return jsonify({
            "error": "Invalid modality"
        }), 400

    path = os.path.join(
        BASE_DIR,
        "static",
        "mri",
        "patient_172",
        f"{modality}.png"
    )

    if not os.path.exists(path):
        return jsonify({
            "error": "MRI image not found"
        }), 404

    return send_file(
        path,
        mimetype="image/png"
    )



if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

# ============================================================
# ADVANCED RESEARCH MODULES (Patient 172 currently validated)
# ============================================================
from flask import send_file

ADVANCED_RESULTS_DIR = BASE_DIR

def _read_first_csv(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    row = df.iloc[0]
    out = {}
    for k, v in row.items():
        if pd.isna(v):
            out[k] = None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out

@app.route('/api/advanced/<int:patient_id>')
def advanced_analysis(patient_id):

    if patient_id != 172:
        return jsonify({
            'patient_id': patient_id,
            'available': False,
            'message': 'Advanced research modules are currently validated for representative Patient 172.'
        })

    return jsonify({
        'patient_id': 172,
        'available': True,

        'quantification': {
            'tumor_voxels': 3534,
            'tumor_slices': 48,
            'max_tumor_slice': 20,
            'relative_burden_percent': 1.6746
        },

        'mc_dropout': {
            'mc_passes': 30,
            'mean_confidence_percent': 85.84,
            'entropy': 0.507,
            'variation_ratio': 0.142,
            'stability': 'Stable'
        },

        'occlusion': {
            'max_confidence_drop': 0.118,
            'influential_slice': 20
        },

        'multi_xai': {
            'gradcam_tumor_attention_percent': 6.75,
            'occlusion_tumor_attention_percent': 8.31,
            'occlusion_tumor_iou': 0.0128,
            'xai_correlation': 0.214
        },

        'similar_cases': [],

        'notes': {
            'mc_interval':
                'Probability spread from stochastic MC-Dropout passes; not a clinical confidence interval.',

            'similarity':
                'Cosine similarity in learned feature space; not medical similarity.',

            'xai':
                'Explanation maps describe model sensitivity and are not tumor segmentations.'
        }
    })

@app.route('/api/advanced-image/<kind>')
def advanced_image(kind):

    paths = {
        'occlusion': os.path.join(
            BASE_DIR,
            'occlusion_sensitivity.png'
        ),

        'multi-xai': os.path.join(
            BASE_DIR,
            'multi_xai_comparison.png'
        ),

        'similar': os.path.join(
            BASE_DIR,
            'similar_patients.png'  
        )
    }

    path = paths.get(kind)

    if not path or not os.path.exists(path):
        return jsonify({
            'error': 'Image not available'
        }), 404

    return send_file(
        path,
        mimetype='image/png'
    )
