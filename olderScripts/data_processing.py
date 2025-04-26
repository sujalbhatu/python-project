import os
import cv2
import numpy as np
import pandas as pd

#— no external dependencies for entropy
def shannon_entropy(gray, num_bins=256):
    if gray is None or gray.size == 0:
        return 0.0
    hist, _ = np.histogram(gray.ravel(),
                           bins=num_bins,
                           range=(0, 256),
                           density=True)
    p = hist[hist > 0]
    return -np.sum(p * np.log2(p))

def fib_sequence(max_val):
    fibs = [1, 1]
    while fibs[-1] < max_val:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

def closest_fib(dim):
    fibs = fib_sequence(dim)
    # pick the fib with minimum |fib - dim|
    return min(fibs, key=lambda x: abs(x - dim))

def preprocess_image(img):
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    # rotate to landscape
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    # resize each side to nearest Fibonacci
    tw = closest_fib(w)
    th = closest_fib(h)
    return cv2.resize(img, (tw, th))

def golden_spiral_segments(img):
    """
    Given img of shape (H, W), carve off 5 rectangular regions by:
      1) vertical slice: width = largest Fibonacci < W, full height
      2) horizontal slice: height = largest Fibonacci < H, full width (of remaining)
      3) vertical slice: width = largest Fibonacci < new W, full height
      4) horizontal slice: height = largest Fibonacci < new H, full width
      5) remainder
    """
    if img is None or img.size == 0:
        return []

    H, W = img.shape[:2]
    segs = []
    x0 = y0 = 0
    cur_w, cur_h = W, H

    # First 4 splits: alternate vertical/horizontal
    for i in range(4):
        if i % 2 == 0:
            # vertical slice
            fibs = fib_sequence(cur_w)
            split = fibs[-2]              # largest Fibonacci < cur_w
            seg = img[y0:y0+cur_h, x0:x0+split]
            segs.append(seg)
            x0    += split
            cur_w -= split

        else:
            # horizontal slice
            fibs = fib_sequence(cur_h)
            split = fibs[-2]              # largest Fibonacci < cur_h
            seg = img[y0:y0+split, x0:x0+cur_w]
            segs.append(seg)
            y0    += split
            cur_h -= split

    # 5th segment: whatever’s left
    segs.append(img[y0:y0+cur_h, x0:x0+cur_w])
    return segs


def compute_region_features(seg, full_entropy, img_path, idx):
    # check emptiness or zero dimension
    if seg is None:
        print(f"  ⚠️ seg #{idx} is None — using zeros")
        return 0.0, 0.0, 0.0, 0.0

    h, w = seg.shape[:2]
    if h == 0 or w == 0:
        print(f"  ⚠️ seg #{idx} has shape ({h},{w}) in {img_path} — using zeros")
        return 0.0, 0.0, 0.0, 0.0

    # grayscale
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY) if seg.ndim == 3 else seg
    m = float(np.mean(gray))
    s = float(np.std(gray))
    e = float(shannon_entropy(gray))
    ig = float(full_entropy - e)
    return m, s, e, ig

def process_folder(base_dir, class_folders):
    rows = []
    for cls in class_folders:
        folder = os.path.join(base_dir, cls)
        if not os.path.isdir(folder):
            print(f"Skipping missing folder: {folder}")
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.png','.jpg','.jpeg')):
                continue

            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            proc = preprocess_image(img)
            if proc is None:
                print(f"Couldn’t preprocess {path}, skipping")
                continue

            # baseline entropy
            gray_full = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            H_full = shannon_entropy(gray_full)

            segs = golden_spiral_segments(proc)
            # if fewer than 5, pad with None
            while len(segs) < 5:
                segs.append(None)

            # for debugging the dimesions

            # print(f"\nProcessing {path}: resized to {proc.shape[:2]}, got {len(segs)} segments")
            # # **Print each segment's dimensions**
            # for i, seg in enumerate(segs[:5], start=1):
            #     if seg is None or seg.size == 0:
            #         print(f"  Segment {i}: empty")
            #     else:
            #         h, w = seg.shape[:2]
            #         print(f"  Segment {i}: {h}×{w}")

            feats = []
            for i, seg in enumerate(segs[:5], start=1):
                m, s, e, ig = compute_region_features(seg, H_full, path, i)
                feats += [m, s, e, ig]

            feats.append(cls)
            rows.append(feats)

    cols = [f'feat{i+1}' for i in range(20)] + ['label']
    df = pd.DataFrame(rows, columns=cols)
    out = os.path.join(base_dir, 'image_features.csv')
    df.to_csv(out, index=False)
    print(f"\nDone: wrote {len(df)} rows to {out}")

if __name__ == '__main__':
    BASE_DIR = os.getcwd()
    CLASS_FOLDERS = ['0', '1', '2']
    process_folder(BASE_DIR, CLASS_FOLDERS)

