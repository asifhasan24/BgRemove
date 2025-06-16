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

def remove_background_mediapipe(image: Image.Image) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = selfie_segmentation.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if results.segmentation_mask is None:
        raise HTTPException(status_code=400, detail="No person detected.")

    mask = results.segmentation_mask > 0.5  # boolean mask

    # Convert the mask to a 3-channel image for element-wise multiplication
    mask_3_channel = np.stack([mask, mask, mask], axis=-1)

    # Create an alpha channel from the mask
    alpha_channel = (mask * 255).astype(np.uint8)

    # Apply the mask to the original image to get the foreground
    foreground = image_np * mask_3_channel

    # Combine the foreground with a transparent background
    # Create an RGBA image. The original image's RGB channels and the new alpha channel.
    output_img_array = np.dstack((foreground, alpha_channel))

    # Convert back to PIL Image
    return Image.fromarray(output_img_array, 'RGBA')

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    try:
        output_img = remove_background_mediapipe(image)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

    buf = io.BytesIO()
    output_img.save(buf, format="PNG") # PNG supports transparency
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
