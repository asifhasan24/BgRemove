from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import mediapipe as mp
import numpy as np
import cv2
import io

app = FastAPI()

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)  # or 0 for ultra-light

def validate_single_face(image: Image.Image):
    # This is optional if you want to keep face detection.
    pass  # Or remove this function if only using segmentation.

def remove_background_mediapipe(image: Image.Image) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = selfie_segmentation.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not results.segmentation_mask.any():
        raise HTTPException(status_code=400, detail="No person detected.")

    mask = results.segmentation_mask > 0.5  # boolean mask

    # Create a white background
    bg_color = 255
    bg_image = np.ones(image_np.shape, dtype=np.uint8) * bg_color

    # Combine image and white background using mask
    output_img = np.where(mask[..., None], image_np, bg_image)

    # Convert back to PIL Image
    return Image.fromarray(output_img)

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    # Optional: validate_single_face(image)

    try:
        output_img = remove_background_mediapipe(image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

    buf = io.BytesIO()
    output_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
