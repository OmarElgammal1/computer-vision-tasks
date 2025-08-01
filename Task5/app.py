import gradio as gr
import torch
import numpy as np
import tifffile
import segmentation_models_pytorch as smp

# --- 1. Load the Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=12,
    classes=1,
)
model.load_state_dict(torch.load("water_segmentation_unet.pth", map_location=torch.device(device)))
model.to(device)
model.eval()

# --- 2. Define the Prediction Function ---
def predict_water_segmentation(image_path):
    """
    Performs inference on a single image, applying the same preprocessing
    as the training dataset.
    """
    # 1. Load image and ensure correct dimension order
    image = tifffile.imread(image_path).astype(np.float32)
    # if image.shape[-1] == 12:
    #     image = np.moveaxis(image, source=-1, destination=0)

    # # 2. Feature Engineering: Add NDWI Band
    # green_band = image[2].copy()
    # nir_band = image[7].copy()
    # epsilon = 1e-6
    # ndwi = (green_band - nir_band) / (green_band + nir_band + epsilon)
    # ndwi = np.expand_dims(ndwi, axis=0)
    # image = np.concatenate([image, ndwi], axis=0)

    # 3. Per-Channel Normalization
    # normalized_image = np.zeros_like(image, dtype=np.float32)
    # for i in range(image.shape[0]):
    #     band = image[i]
    #     min_val, max_val = band.min(), band.max()
    #     if max_val > min_val:
    #         normalized_image[i] = (band - min_val) / (max_val - min_val)
    #     else:
    #         normalized_image[i] = band

    # Convert to tensor and add batch dimension
    input_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    # 4. Perform prediction
    with torch.no_grad():
        logits = model(input_tensor)

    # 5. Post-process the output
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8) * 255

    return preds

# --- 3. Create the Gradio Interface ---
iface = gr.Interface(
    fn=predict_water_segmentation,
    inputs=gr.File(label="Upload a .tif Image"),
    outputs=gr.Image(label="Segmented Water Mask"),
    title="Water Segmentation Model",
    description="Upload a 12-channel .tif image to see the water segmentation mask.",
)

iface.launch()