# flake8: noqa
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np
import cv2
from final_model import FinalModel
import torchvision.transforms as transforms

def brush_callback(event, x, y, flags, user_data):
    mask = user_data["mask"]
    drawing = user_data["drawing"]
    brush_radius = user_data["brush_radius"]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing[0] = True
        cv2.circle(mask, (x, y), brush_radius, 255, -1)

    elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
        cv2.circle(mask, (x, y), brush_radius, 255, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing[0] = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        user_data["finished"][0] = True

def postprocess(image):
    image = image * 255.0
    image = image.permute(1, 2, 0)
    return image.int()

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
    )

    if not file_path:
        print("No file selected.")
        return
    
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])

    image = Image.open(file_path)
    image = transform(image)

    image_np = np.array(postprocess(image)).astype('uint8')
    image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
    drawing = [False]
    brush_radius = 5
    finished = [False]

    original_image = image_np_rgb.copy()

    user_data = {
        "mask": mask,
        "drawing": drawing,
        "brush_radius": brush_radius,
        "finished": finished
    }
    cv2.namedWindow("Draw mask")
    cv2.setMouseCallback("Draw mask", brush_callback, user_data)

    while not finished[0]:
        display_image = original_image.copy()
        display_image[mask == 255] = (0, 255, 0)
        cv2.imshow("Draw mask", display_image)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    if mask is not None:
        model = FinalModel()
        model.complete_image(image, mask)
    else:
        print("No mask was created.")

if __name__ == "__main__":
    main()