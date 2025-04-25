import cv2
import numpy as np
import matplotlib.pyplot as plt

def fib_sequence(max_val):
    f = [1, 1]
    while f[-1] < max_val:
        f.append(f[-1] + f[-2])
    return f

def preprocess_image(img):
    h, w = img.shape[:2]
    # rotate to landscape
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    # resize each side to nearest Fibonacci
    tw = min(fib_sequence(w), key=lambda x: abs(x - w))
    th = min(fib_sequence(h), key=lambda x: abs(x - h))
    return cv2.resize(img, (tw, th))

def golden_spiral_coords(img):
    H, W = img.shape[:2]
    coords = []
    x0 = y0 = 0
    cur_w, cur_h = W, H

    # 4 alternate splits
    for i in range(4):
        if i % 2 == 0:
            # vertical
            f = fib_sequence(cur_w)
            split = f[-2]
            coords.append((x0, y0, split, cur_h))
            x0    += split
            cur_w -= split
        else:
            # horizontal
            f = fib_sequence(cur_h)
            split = f[-2]
            coords.append((x0, y0, cur_w, split))
            y0    += split
            cur_h -= split

    # final remainder
    coords.append((x0, y0, cur_w, cur_h))
    return coords

def visualize_spiral(img_path, draw_arcs=False):
    img = cv2.imread(img_path)
    proc = preprocess_image(img)
    coords = golden_spiral_coords(proc)

    # the “processing” order is [0,1,2,3,4], but we want to label in
    # spiral (clockwise) order: 0→1→4→2→3
    spiral_order = [0, 1, 4, 2, 3]
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]

    disp = proc.copy()
    for lab_idx, seg_idx in enumerate(spiral_order, start=1):
        x,y,w,h = coords[seg_idx]
        c = colors[lab_idx-1]
        # rectangle
        cv2.rectangle(disp, (x,y), (x+w, y+h), c, 2)
        # label
        cv2.putText(disp, str(lab_idx), (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)

        # optional: draw quarter-circle in each square region
        if draw_arcs and w==h:
            # decide which corner to anchor arc
            if seg_idx in (0, 3):  # top‐left and bottom‐left
                center = (x+w, y+h)
                startAng, endAng = 180, 270
            elif seg_idx in (1, 4):  # top‐right and bottom‐right
                center = (x, y+h)
                startAng, endAng = 270, 360
            else:                  # the middle region
                center = (x, y)
                startAng, endAng =   0, 90
            cv2.ellipse(disp, center, (w, h), 0,
                        startAng, endAng, c, 2)

    # show via matplotlib
    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,8))
    plt.imshow(rgb)
    plt.axis('off')
    plt.title("Golden‐Spiral Segments (1→2→3→4→5)")
    plt.show()

if __name__ == '__main__':
    visualize_spiral('1/10_augmented_2.png', draw_arcs=False)
