import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def fib_sequence(n):
    f = [1, 1]
    while f[-1] < n:
        f.append(f[-1] + f[-2])
    return f

def closest_fib(x):
    fs = fib_sequence(x)
    return min(fs, key=lambda v: abs(v - x))

def preprocess_image(img):
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    tw, th = closest_fib(w), closest_fib(h)
    return cv2.resize(img, (tw, th))

def golden_spiral_segments_inward(img):
    H, W = img.shape[:2]
    segs, coords = [], []
    x0, y0, cw, ch = 0, 0, W, H

    # 1) LEFT big slice
    split = fib_sequence(cw)[-2]
    coords.append((x0, y0, split, ch))
    segs.append(img[y0:y0+ch, x0:x0+split])
    x0 += split; cw -= split

    # 2) TOP big slice
    split = fib_sequence(ch)[-2]
    coords.append((x0, y0, cw, split))
    segs.append(img[y0:y0+split, x0:x0+cw])
    y0 += split; ch -= split

    # 3) RIGHT big slice
    split = fib_sequence(cw)[-2]
    coords.append((x0+cw-split, y0, split, ch))
    segs.append(img[y0:y0+ch, x0+cw-split:x0+cw])
    cw -= split

    # 4) BOTTOM big slice
    split = fib_sequence(ch)[-2]
    coords.append((x0, y0+ch-split, cw, split))
    segs.append(img[y0+ch-split:y0+ch, x0:x0+cw])
    ch -= split

    # 5) CENTER
    coords.append((x0, y0, cw, ch))
    segs.append(img[y0:y0+ch, x0:x0+cw])

    return coords

def draw_inward_segments(img, coords):
    disp = img.copy()
    colors = [
        (255, 0,   0),
        (0,   255, 0),
        (0,   0,   255),
        (255, 255, 0),
        (255, 0,   255),
    ]
    for i, (x, y, w, h) in enumerate(coords, start=1):
        c = colors[i-1]
        cv2.rectangle(disp, (x, y), (x+w, y+h), c, 2)
        cv2.putText(disp, str(i), (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return disp

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='Path to input image')
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("❌ Could not load image:", args.image)
        return

    proc = preprocess_image(img)
    coords = golden_spiral_segments_inward(proc)
    vis = draw_inward_segments(proc, coords)

    # display
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Inward Spiral Segmentation (1→5)')
    plt.show()

if __name__ == '__main__':
    main()
