import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# finding the fibonacci sequence till nth number
def fib_sequence(n):
    f = [1, 1]
    while f[-1] < n:
        f.append(f[-1] + f[-2])
    return f

def closest_fib(x):
    fs = fib_sequence(x)
    # find the fibonacci number cloeset to x
    # but can it do the job using abs ??? wouldn't it use abs(x-fib)? it x can be greater too

    # which way would be better , losing the data (info pixels) or add pixels that don't make sense
    return min(fs, key=lambda v: abs(v - x))

def preprocess_image(img):
    # we get the image , and from that we get the hieght and width
    # img.shape is a tuple (height, width, channels)
    # hence we take until to , to have height and width
    h, w = img.shape[:2]
    # rotate to landscape if it is in portrait
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    
    # resize each side to nearest Fibonacci so that we can put the golden ratio thing
    tw, th = closest_fib(w), closest_fib(h)
    return cv2.resize(img, (tw, th))
    
def golden_spiral_segments_inward(img):
    H, W = img.shape[:2]
    segs, coords = [], []
    # x0 and y0 are the top left corner , we do calculations with respect to that
    x0, y0, cw, ch = 0, 0, W, H

    # i send current width , so that i get the largest fib < current width
    split = fib_sequence(cw)[-2]

    # store the coordinatees of the segments
    coords.append((x0, y0, split, ch))
    # store the image of the segment, the chutku left part
    segs.append(img[y0:y0+ch, x0:x0+split])
    # we are moving the top left corner to the right by the split,
    # and then after that we have to reduce the current width by the split
    x0 += split; cw -= split

    # now we do the same thing but for the horizontal split
    # we now onward repeated it in anticlockwise direction

    split = fib_sequence(ch)[-2]
    coords.append((x0, y0, cw, split))
    segs.append(img[y0:y0+split, x0:x0+cw])
    y0 += split; ch -= split

    split = fib_sequence(cw)[-2]
    coords.append((x0+cw-split, y0, split, ch))
    segs.append(img[y0:y0+ch, x0+cw-split:x0+cw])
    cw -= split

    split = fib_sequence(ch)[-2]
    coords.append((x0, y0+ch-split, cw, split))
    segs.append(img[y0+ch-split:y0+ch, x0:x0+cw])
    ch -= split

    # at last we just are left with the last segment , 
    coords.append((x0, y0, cw, ch))
    # saving the image of the last segment
    segs.append(img[y0:y0+ch, x0:x0+cw])

    # finally we just return the the coordinates , 

    # we have made image segments too , but we have not exported here
    return coords

def draw_inward_segments(img, coords):
    # we dont need segments , what we needed was just the coordinates so that we 
    # can draw the rectangles
    #creating copy
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
    # main file to run
    p = argparse.ArgumentParser()
    # path to the image
    # to take image from the image
    p.add_argument('--image', required=True, help='Path to input image')
    args = p.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("Could not load image:", args.image)
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
