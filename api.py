
# api.py
import os
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# ------------------------- CONFIG -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE1_MODEL_PATH = "stage1_traced_final.pt"
STAGE2_MODEL_PATH = "stage2_traced_final.pt"

STAGE1_CLASSES = ["no_trash", "trash"]
STAGE2_CLASSES = ["small", "medium", "large"]

# ------------------------- LOAD MODELS -------------------------
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = torch.jit.load(model_path, map_location=DEVICE)
    model.eval()
    return model

stage1_model = load_model(STAGE1_MODEL_PATH)
stage2_model = load_model(STAGE2_MODEL_PATH)

# ------------------------- TRANSFORMS -------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------------- HELPERS -------------------------
def predict_image(model, image: Image.Image, classes):
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)
    return classes[idx.item()], float(conf.item())

def compare_image_locations(img1_bytes: bytes, img2_bytes: bytes, min_good_matches=10):
    npimg1 = np.frombuffer(img1_bytes, np.uint8)
    npimg2 = np.frombuffer(img2_bytes, np.uint8)
    img1 = cv2.imdecode(npimg1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(npimg2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return "Error Loading Images"

    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return "Different Location"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    if matches and all(len(m) == 2 for m in matches):
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    return "Same Location" if len(good_matches) > min_good_matches else "Different Location"

# ------------------------- FASTAPI APP -------------------------
app = FastAPI(title="Waste Management Pipeline API")

@app.get("/")
def root():
    return {"message": "Waste management API is running!"}

# ------------------------- STAGE 1 -------------------------
@app.post("/predict_stage1")
async def predict_stage1(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        prediction, confidence = predict_image(stage1_model, image, STAGE1_CLASSES)
        return JSONResponse({
            "filename": file.filename,
            "stage": "Stage 1",
            "prediction": prediction,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------- STAGE 2 -------------------------
@app.post("/predict_stage2")
async def predict_stage2(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        prediction, confidence = predict_image(stage2_model, image, STAGE2_CLASSES)
        return JSONResponse({
            "filename": file.filename,
            "stage": "Stage 2",
            "prediction": prediction,
            "confidence": confidence
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------- FULL PIPELINE -------------------------
@app.post("/pipeline")
async def run_pipeline(
    primary_image: UploadFile = File(...),
    secondary_image: UploadFile = File(...)
):
    try:
        primary_bytes = await primary_image.read()
        secondary_bytes = await secondary_image.read()

        # Stage 1: Trash Detection
        pil_primary = Image.open(BytesIO(primary_bytes)).convert("RGB")
        stage1_pred, stage1_conf = predict_image(stage1_model, pil_primary, STAGE1_CLASSES)

        if stage1_pred == "trash":
            # Stage 2 (size classifier) if needed
            stage2_pred, stage2_conf = predict_image(stage2_model, pil_primary, STAGE2_CLASSES)
            final_result = f"Trash Detected - Size: {stage2_pred}"
            return JSONResponse({
                "primary_image": primary_image.filename,
                "stage1_prediction": stage1_pred,
                "stage1_confidence": stage1_conf,
                "stage2_prediction": stage2_pred,
                "stage2_confidence": stage2_conf,
                "final_result": final_result
            })

        # Stage 2: Location Comparison
        location_result = compare_image_locations(primary_bytes, secondary_bytes)
        final_result = f"No Trash, {location_result}"

        return JSONResponse({
            "primary_image": primary_image.filename,
            "secondary_image": secondary_image.filename,
            "stage1_prediction": stage1_pred,
            "stage1_confidence": stage1_conf,
            "stage2_result": location_result,
            "final_result": final_result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
