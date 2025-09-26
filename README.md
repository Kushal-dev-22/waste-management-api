
2️⃣ POST /predict_stage2
Classifies the amount of trash in the image.
Input: file (image: JPG, JPEG, PNG)
Response:
json{
  "trash_amount": "medium",
  "confidence": 0.78,
  "categories": {
    "small": 0.15,
    "medium": 0.78,
    "large": 0.07
  }
}
3️⃣ POST /pipeline
Compares two images to check if trash was removed and if photos are from the same scene.
Input:

primary_image (after image)
secondary_image (before image)

Response:
json{
  "trash_removed": true,
  "same_scene": true,
  "confidence_trash_removal": 0.82,
  "confidence_scene_match": 0.91
}
