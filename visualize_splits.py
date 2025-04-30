import sys
import cv2
from final_working import preprocess_image, golden_spiral_segments_inward

def visualize(path, out=None):
    img = cv2.imread(path)
    if img is None:
        print("Can't load", path); return
    proc = preprocess_image(img)
    # get only the coords, ignore the segment arrays
    _, coords = golden_spiral_segments_inward(cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY))
    vis = proc.copy()
    for (x,y,w,h) in coords:
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
    if out:
        cv2.imwrite(out, vis)
        print("Saved vis to", out)
    cv2.imshow("Fibonacci Spiral Split", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_splits.py <input_image> [output_image]")
    else:
        inp = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else None
        visualize(inp, out)
