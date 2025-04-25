import cv2
import numpy as np
import matplotlib.pyplot as plt

# Use previously defined functions or re-import them
# If defined in another module, adjust the import accordingly
# from your_module import fib_sequence, preprocess_image

def fib_sequence(max_val):
    fibs = [1, 1]
    while fibs[-1] < max_val:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def preprocess_image(img):
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    tw = min(fib_sequence(w), key=lambda x: abs(x - w))
    th = min(fib_sequence(h), key=lambda x: abs(x - h))
    return cv2.resize(img, (tw, th))


def golden_spiral_coords(img):
    """
    Returns a list of (x, y, w, h) tuples for the 5 golden spiral segments
    """
    H, W = img.shape[:2]
    coords = []
    x0 = y0 = 0
    cur_w, cur_h = W, H

    for i in range(4):
        if i % 2 == 0:
            fibs = fib_sequence(cur_w)
            split = fibs[-2]  # largest Fibonacci < cur_w
            coords.append((x0, y0, split, cur_h))
            x0 += split
            cur_w -= split
        else:
            fibs = fib_sequence(cur_h)
            split = fibs[-2]  # largest Fibonacci < cur_h
            coords.append((x0, y0, cur_w, split))
            y0 += split
            cur_h -= split

    # Final remainder segment
    coords.append((x0, y0, cur_w, cur_h))
    return coords


def visualize_segments(img_path):
    img = cv2.imread(img_path)
    proc = preprocess_image(img)
    if proc is None:
        print(f"Couldn\'t preprocess {img_path}")
        return

    coords = golden_spiral_coords(proc)
    img_disp = proc.copy()
    # Colors in BGR
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

    for idx, (x, y, w, h) in enumerate(coords, start=1):
        color = colors[(idx - 1) % len(colors)]
        cv2.rectangle(img_disp, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_disp, str(idx), (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title("Golden Spiral Segments")
    plt.show()


if __name__ == '__main__':
    # Replace 'path/to/image.jpg' with your image file
    visualize_segments('1/10_augmented_2.png')
