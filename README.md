# 🗑️ Waste Management API

A machine learning API for trash detection and monitoring built with PyTorch and MobileNet v2.

**🌐 Live API**: https://waste-management-api-hkfz.onrender.com/

## ✨ Features

- 🔍 **Trash Detection**: Identify if an image contains trash
- 📊 **Trash Classification**: Classify trash amount (small/medium/large)
- 🔄 **Scene Comparison**: Compare before/after images to verify cleanup

## 🧠 Model Architecture

Built using **PyTorch** ⚡ with **MobileNet v2** 📱 - a lightweight architecture optimized for efficient inference while maintaining good accuracy.

## 🚀 API Endpoints

### 1️⃣ POST `/predict_stage1`
Detects whether an image contains trash.

**Input**: `file` (image: JPG, JPEG, PNG)

**Response**:
```json
{
  "filename": "images204.jpg",
  "stage": "Stage 1",
  "prediction": "trash",
  "confidence": 0.9980435371398926
}
```

### 2️⃣ POST `/predict_stage2`
Classifies the amount of trash in the image.

**Input**: `file` (image: JPG, JPEG, PNG)

**Response**:
```json
{
  "filename": "95406aa8.jpg",
  "stage": "Stage 2",
  "prediction": "large",
  "confidence": 0.8675705552101135
}
```

### 3️⃣ POST `/pipeline`
Compares two images to check if trash was removed and if photos are from the same scene.

**Input**: 
- `primary_image` (after image)
- `secondary_image` (before image)

**Response**:
```json
{
  "primary_image": "lavagem_de.jpg",
  "secondary_image": "images17_jpg",
  "stage1_prediction": "no_trash",
  "stage1_confidence": 0.9407783150672913,
  "stage2_result": "Different Location",
  "final_result": "No Trash, Different Location"
}
```
