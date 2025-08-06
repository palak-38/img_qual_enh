import cv2
import torch
import numpy as np

def stitch_patches(patches, patch_coords, out_h, out_w):
    stitched_image = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w, 3), dtype=np.float32)
    for (patch, (top, left)) in zip(patches, patch_coords):
        ph, pw, _ = patch.shape
        stitched_image[top:top + ph, left:left + pw, :] += patch
        weight_map[top:top + ph, left:left + pw, :] += 1.0
    stitched_image /= np.maximum(weight_map, 1e-8)
    return stitched_image

def img_to_tensor(img):
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img.transpose(2, 0, 1))
    return tensor

def inference_large_image(model_path, input_image_path, output_image_path, scale_factor=2, patch_size=128, overlap=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleSRNet(scale_factor=scale_factor)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    lr_bgr = cv2.imread(input_image_path)
    if lr_bgr is None:
        raise ValueError(f"Could not read the image from {input_image_path}")
    lr_rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
    lr_h, lr_w, _ = lr_rgb.shape
    hr_h, hr_w = lr_h * scale_factor, lr_w * scale_factor
    patches = []
    patch_coords = []
    step = patch_size - overlap
    for top in range(0, lr_h, step):
        for left in range(0, lr_w, step):
            bottom = min(top + patch_size, lr_h)
            right = min(left + patch_size, lr_w)
            lr_patch = lr_rgb[top:bottom, left:right, :]
            lr_patch_t = img_to_tensor(lr_patch).unsqueeze(0).to(device)
            with torch.no_grad():
                sr_patch_t = model(lr_patch_t)
            sr_patch = sr_patch_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            sr_patch = np.clip(sr_patch, 0.0, 1.0)
            patches.append(sr_patch)
            patch_coords.append((top * scale_factor, left * scale_factor))
    sr_image = stitch_patches(patches, patch_coords, hr_h, hr_w)
    sr_image_8u = (sr_image * 255.0).astype(np.uint8)
    result_bgr = cv2.cvtColor(sr_image_8u, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_image_path, result_bgr)
    print(f"Enhanced image saved to {output_image_path}")

if __name__ == "__main__":
    model_path = "sr_model.pth"
    input_image_path = "input_image.png" #replace with input image path
    output_image_path = "enhanced_image.png"
    inference_large_image(model_path, input_image_path, output_image_path)
