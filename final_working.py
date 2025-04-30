import cv2
import numpy as np
import os
import csv
from scipy.stats import entropy

def fib_sequence(n):
    f = [1, 1]
    while f[-1] < n:
        f.append(f[-1] + f[-2])
    return f

# Return largest Fibonacci <= x (for preprocess)
def floor_fib(x):
    f = [v for v in fib_sequence(x) if v <= x]
    return max(f) if f else 1

# Return largest Fibonacci < x (for segmentation)
def prev_fib(x):
    f = [v for v in fib_sequence(x) if v < x]
    return max(f) if f else 1


# --- Preprocessing ---
def preprocess_image(img):
    """Rotate if needed and resize to nearest Fibonacci dimensions."""
    h, w = img.shape[:2]
    # ensure width >= height
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    # crop/rescale using floor Fibonacci
    tw = floor_fib(w)
    th = floor_fib(h)
    return cv2.resize(img, (tw, th))

# --- Segmentation via inward golden spiral ---
def golden_spiral_segments_inward(img):
    """Split image into 5 segments following an inward Fibonacci spiral."""
    H, W = img.shape[:2]
    segs, coords = [], []
    x0, y0, cw, ch = 0, 0, W, H

    # 1) left vertical
    split = prev_fib(cw)
    coords.append((x0, y0, split, ch))
    segs.append(img[y0:y0+ch, x0:x0+split])
    x0 += split; cw -= split

    # 2) top horizontal
    split = prev_fib(ch)
    coords.append((x0, y0, cw, split))
    segs.append(img[y0:y0+split, x0:x0+cw])
    y0 += split; ch -= split

    # 3) right vertical
    split = prev_fib(cw)
    coords.append((x0+cw-split, y0, split, ch))
    segs.append(img[y0:y0+ch, x0+cw-split:x0+cw])
    cw -= split

    # 4) bottom horizontal
    split = prev_fib(ch)
    coords.append((x0, y0+ch-split, cw, split))
    segs.append(img[y0+ch-split:y0+ch, x0:x0+cw])
    ch -= split

    # 5) center
    coords.append((x0, y0, cw, ch))
    segs.append(img[y0:y0+ch, x0:x0+cw])

    return segs, coords

# --- Metrics ---
def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
    hist /= hist.sum()
    return entropy(hist, base=2)

def calculate_info_gain(parent_entropy, segment_entropy):
    return parent_entropy - segment_entropy

# --- Feature Extraction ---
def extract_features(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    segments, _ = golden_spiral_segments_inward(img_gray)
    parent_entropy = calculate_entropy(img_gray)

    features = []
    for seg in segments:
        mean_val = np.mean(seg)
        stddev_val = np.std(seg)
        ent = calculate_entropy(seg)
        info_gain = calculate_info_gain(parent_entropy, ent)
        features.extend([mean_val, stddev_val, ent, info_gain])
    return features

# --- Processing Function ---
def process_dataset(input_dir='input_images', output_csv='features.csv', output_seg_dir='segmented_images'):
    os.makedirs(output_seg_dir, exist_ok=True)
    data = []
    header = []

    # Build CSV header
    for i in range(5):
        header.extend([
            f'segment_{i+1}_mean',
            f'segment_{i+1}_stddev',
            f'segment_{i+1}_entropy',
            f'segment_{i+1}_info_gain'
        ])
    header.append('label')

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[!] Could not load image: {img_path}")
                    continue
                label = os.path.basename(os.path.dirname(img_path))
                print(f"[✓] Processing {filename} (label: {label})...")

                img_proc = preprocess_image(img)
                feature_vector = extract_features(img_proc)
                feature_vector.append(label)
                data.append(feature_vector)

                # Save segments
                save_folder = os.path.join(output_seg_dir, os.path.splitext(filename)[0])
                os.makedirs(save_folder, exist_ok=True)
                segments, coords = golden_spiral_segments_inward(img_proc)
                for idx, seg in enumerate(segments):
                    seg_path = os.path.join(save_folder, f'segment_{idx+1}.png')
                    cv2.imwrite(seg_path, seg)

    # Write CSV
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    print(f"\nFeature extraction and segmentation complete.")
    print(f"Features saved to '{output_csv}'")
    print(f"Segmented images saved under '{output_seg_dir}'")

# --- Main ---
if __name__ == '__main__':
    process_dataset()
