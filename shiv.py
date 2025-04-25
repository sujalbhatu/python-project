import os
import cv2
import numpy as np
import pandas as pd

# Shannon entropy (no external deps)
def shannon_entropy(gray, num_bins=256):
    if gray is None or gray.size==0:
        return 0.0
    hist,_ = np.histogram(gray.ravel(), bins=num_bins, range=(0,256), density=True)
    p = hist[hist>0]
    return -np.sum(p * np.log2(p))

# Fibonacci
def fib_sequence(n):
    f=[1,1]
    while f[-1]<n:
        f.append(f[-1]+f[-2])
    return f

def closest_fib(x):
    fs = fib_sequence(x)
    return min(fs, key=lambda v: abs(v-x))

# Preprocess: ensure landscape + resize to nearest fib dims
def preprocess_image(img):
    if img is None or img.size==0:
        return None
    h,w = img.shape[:2]
    if h>w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h,w = img.shape[:2]
    return cv2.resize(img, (closest_fib(w), closest_fib(h)))

# Compute rectangle coords for golden spiral splits
def golden_spiral_coords(img):
    H,W = img.shape[:2]
    coords=[]
    x0=y0=0
    cw,ch = W,H
    for i in range(4):
        if i%2==0:
            split = fib_sequence(cw)[-2]
            coords.append((x0,y0,split,ch))
            x0 += split; cw -= split
        else:
            split = fib_sequence(ch)[-2]
            coords.append((x0,y0,cw,split))
            y0 += split; ch -= split
    coords.append((x0,y0,cw,ch))
    return coords

# Draw colored boxes + labels in spiral order
def draw_segments(img, coords):
    disp = img.copy()
    colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255)]
    spiral = [0,1,4,2,3]
    for lbl, idx in enumerate(spiral,1):
        x,y,w,h = coords[idx]
        c = colors[lbl-1]
        cv2.rectangle(disp, (x,y), (x+w,y+h), c, 2)
        cv2.putText(disp, str(lbl), (x+5,y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, c, 2)
    return disp

# Extract one region's features
def compute_region_features(seg, H_full, idx, path):
    if seg is None or seg.size==0:
        print(f"  ⚠ seg#{idx} empty @ {path}")
        return (0.0,0.0,0.0,0.0)
    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY) if seg.ndim==3 else seg
    m = float(np.mean(gray))
    s = float(np.std(gray))
    e = float(shannon_entropy(gray))
    ig = float(H_full - e)
    return (m,s,e,ig)

# Main processing + visualization
def process_folder(base_dir, class_folders):
    rows=[]
    vis_dir = os.path.join(base_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    for cls in class_folders:
        fld = os.path.join(base_dir, cls)
        if not os.path.isdir(fld):
            continue
        for fn in os.listdir(fld):
            if not fn.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            path = os.path.join(fld, fn)
            img = cv2.imread(path)
            proc = preprocess_image(img)
            if proc is None:
                continue

            # baseline entropy
            gray_full = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            H_full = shannon_entropy(gray_full)

            # coords & spiral-ordered segments
            coords = golden_spiral_coords(proc)
            spiral = [0,1,4,2,3]
            segs = [ proc[y:y+h, x:x+w] for idx in spiral
                     for (x,y,w,h) in [coords[idx]] ]

            # pad in case <5
            segs += [None] * (5 - len(segs))

            # extract features
            feats=[]
            for i,sg in enumerate(segs,1):
                feats += list(compute_region_features(sg, H_full, i, path))
            feats.append(cls)
            rows.append(feats)

            # save visualization overlay
            vis_img = draw_segments(proc, coords)
            out_vis = os.path.join(vis_dir, f"vis_{cls}_{fn}")
            cv2.imwrite(out_vis, vis_img)

    # write CSV
    cols = [f'feat{i+1}' for i in range(20)] + ['label']
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(os.path.join(base_dir,'image_features.csv'), index=False)
    print(f"✅ Done: {len(rows)} rows + visualizations in {vis_dir}")

if __name__=='__main__':
    BASE_DIR = os.getcwd()
    CLASS_FOLDERS = ['0','1','2']
    process_folder(BASE_DIR, CLASS_FOLDERS)
