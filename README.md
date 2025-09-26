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
  "has_trash": true,
  "confidence": 0.85,
  "message": "Trash detected in the image"
}
```

### 2️⃣ POST `/predict_stage2`
Classifies the amount of trash in the image.

**Input**: `file` (image: JPG, JPEG, PNG)

**Response**:
```json
{
  "trash_amount": "medium",
  "confidence": 0.78,
  "categories": {
    "small": 0.15,
    "medium": 0.78,
    "large": 0.07
  }
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
  "trash_removed": true,
  "same_scene": true,
  "confidence_trash_removal": 0.82,
  "confidence_scene_match": 0.91
}
```
