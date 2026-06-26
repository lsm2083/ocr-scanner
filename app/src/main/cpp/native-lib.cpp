#include <jni.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

// Keep in sync with MainActivity.Mode ordinals.
enum Mode {
    MODE_ORIGINAL = 0,
    MODE_GRAY = 1,
    MODE_BLUR = 2,
    MODE_CANNY = 3,
    MODE_CANNY_OVERLAY = 4,
    MODE_SKETCH = 5,
    MODE_SHAPES = 6,
    MODE_DOC = 7
};

static const Scalar GREEN(0, 255, 0, 255);
static const Scalar RED(255, 60, 60, 255);
static const Scalar BLUE(60, 160, 255, 255);
static const Scalar YELLOW(255, 220, 0, 255);

// ---- helpers ---------------------------------------------------------------

static void drawLabel(Mat &img, const string &text, Point org, const Scalar &color,
                      double scale) {
    putText(img, text, org, FONT_HERSHEY_SIMPLEX, scale, Scalar(0, 0, 0, 255),
            (int) (scale * 6) + 3, LINE_AA);
    putText(img, text, org, FONT_HERSHEY_SIMPLEX, scale, color,
            (int) (scale * 3) + 1, LINE_AA);
}

// Pencil-sketch via the classic color-dodge of gray over its inverted blur.
static void pencilSketch(const Mat &gray, Mat &dst) {
    Mat inv, blur, invBlur;
    bitwise_not(gray, inv);
    GaussianBlur(inv, blur, Size(21, 21), 0);
    bitwise_not(blur, invBlur);
    divide(gray, invBlur, dst, 256.0);
}

// Finds the largest convex 4-point contour (a document candidate). Empty if none.
static vector<Point> findDocQuad(const Mat &rgba, double t1, double t2) {
    Mat gray, edges;
    cvtColor(rgba, gray, COLOR_RGBA2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);
    Canny(gray, edges, t1, t2);
    dilate(edges, edges, getStructuringElement(MORPH_RECT, Size(5, 5)));

    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double imgArea = (double) rgba.cols * rgba.rows;
    double bestArea = 0;
    vector<Point> best;
    for (auto &c: contours) {
        double a = contourArea(c);
        if (a < 0.10 * imgArea) continue;
        vector<Point> approx;
        approxPolyDP(c, approx, 0.02 * arcLength(c, true), true);
        if (approx.size() == 4 && isContourConvex(approx) && a > bestArea) {
            bestArea = a;
            best = approx;
        }
    }
    return best;
}

// Orders 4 points as TL, TR, BR, BL.
static void orderPoints(const vector<Point> &pts, Point2f out[4]) {
    int tl = 0, br = 0, tr = 0, bl = 0;
    double minSum = 1e18, maxSum = -1e18, minDiff = 1e18, maxDiff = -1e18;
    for (int i = 0; i < 4; i++) {
        double sum = pts[i].x + pts[i].y;
        double diff = pts[i].x - pts[i].y;
        if (sum < minSum) { minSum = sum; tl = i; }
        if (sum > maxSum) { maxSum = sum; br = i; }
        if (diff > maxDiff) { maxDiff = diff; tr = i; }
        if (diff < minDiff) { minDiff = diff; bl = i; }
    }
    out[0] = pts[tl];
    out[1] = pts[tr];
    out[2] = pts[br];
    out[3] = pts[bl];
}

// Detects contours, classifies shapes, draws boxes + per-frame object count.
static void detectShapes(const Mat &rgba, Mat &out, double t1, double t2) {
    Mat gray, edges;
    cvtColor(rgba, gray, COLOR_RGBA2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 1.2);
    Canny(gray, edges, t1, t2);
    dilate(edges, edges, getStructuringElement(MORPH_RECT, Size(3, 3)));

    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    rgba.copyTo(out);
    double imgArea = (double) rgba.cols * rgba.rows;
    double scale = rgba.rows / 720.0 * 0.7;
    int count = 0;

    for (auto &c: contours) {
        double a = contourArea(c);
        if (a < 0.004 * imgArea) continue;  // ignore noise

        vector<Point> approx;
        approxPolyDP(c, approx, 0.03 * arcLength(c, true), true);

        string name;
        Scalar color;
        int v = (int) approx.size();
        if (v == 3) {
            name = "Tri"; color = RED;
        } else if (v == 4) {
            Rect r = boundingRect(approx);
            double ar = (double) r.width / r.height;
            name = (ar >= 0.85 && ar <= 1.15) ? "Square" : "Rect";
            color = GREEN;
        } else if (v == 5) {
            name = "Penta"; color = YELLOW;
        } else {
            // Many vertices + high circularity => circle.
            double peri = arcLength(c, true);
            double circularity = 4 * CV_PI * a / (peri * peri);
            if (circularity > 0.80) { name = "Circle"; color = BLUE; }
            else { name = "Poly"; color = YELLOW; }
        }

        Rect box = boundingRect(approx);
        rectangle(out, box, color, max(2, (int) (scale * 3)));
        drawLabel(out, name, Point(box.x, box.y - 6), color, scale);
        count++;
    }

    drawLabel(out, "Objects: " + to_string(count), Point(12, (int) (40 * scale) + 10),
              GREEN, scale * 1.2);
}

// Boosts text legibility for OCR: upscales small inputs, then grayscale +
// adaptive contrast (CLAHE) + unsharp mask. Keeps it natural (no hard threshold).
static void enhanceForOcr(Mat &img, Mat &out) {
    Mat work;
    double longSide = max(img.cols, img.rows);
    if (longSide < 2000.0) {
        double s = 2000.0 / longSide;
        resize(img, work, Size(), s, s, INTER_CUBIC);
    } else {
        work = img;
    }
    Mat gray, blurred;
    cvtColor(work, gray, COLOR_RGBA2GRAY);
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(gray, gray);
    GaussianBlur(gray, blurred, Size(0, 0), 1.0);
    addWeighted(gray, 1.5, blurred, -0.5, 0, gray);
    cvtColor(gray, out, COLOR_GRAY2RGBA);
}

// ---- JNI -------------------------------------------------------------------

extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_detectapp_MainActivity_nativeProcess(
        JNIEnv * /*env*/, jobject /*thiz*/,
        jlong inAddr, jlong outAddr, jint mode, jdouble t1, jdouble t2) {

    Mat &rgba = *reinterpret_cast<Mat *>(inAddr);
    Mat &out = *reinterpret_cast<Mat *>(outAddr);

    auto start = chrono::high_resolution_clock::now();

    Mat gray, edges, tmp;
    switch (mode) {
        case MODE_ORIGINAL:
            rgba.copyTo(out);
            break;
        case MODE_GRAY:
            cvtColor(rgba, gray, COLOR_RGBA2GRAY);
            cvtColor(gray, out, COLOR_GRAY2RGBA);
            break;
        case MODE_BLUR:
            GaussianBlur(rgba, out, Size(15, 15), 0);
            break;
        case MODE_CANNY:
            cvtColor(rgba, gray, COLOR_RGBA2GRAY);
            GaussianBlur(gray, gray, Size(5, 5), 1.4);
            Canny(gray, edges, t1, t2);
            cvtColor(edges, out, COLOR_GRAY2RGBA);
            break;
        case MODE_CANNY_OVERLAY:
            cvtColor(rgba, gray, COLOR_RGBA2GRAY);
            GaussianBlur(gray, gray, Size(5, 5), 1.4);
            Canny(gray, edges, t1, t2);
            rgba.copyTo(out);
            out.setTo(GREEN, edges);
            break;
        case MODE_SKETCH:
            cvtColor(rgba, gray, COLOR_RGBA2GRAY);
            pencilSketch(gray, tmp);
            cvtColor(tmp, out, COLOR_GRAY2RGBA);
            break;
        case MODE_SHAPES:
            detectShapes(rgba, out, t1, t2);
            break;
        case MODE_DOC: {
            rgba.copyTo(out);
            vector<Point> quad = findDocQuad(rgba, t1, t2);
            double scale = rgba.rows / 720.0 * 0.7;
            if (quad.size() == 4) {
                vector<vector<Point>> polys{quad};
                polylines(out, polys, true, GREEN, max(3, (int) (scale * 4)), LINE_AA);
                for (auto &p: quad)
                    circle(out, p, max(6, (int) (scale * 8)), RED, FILLED, LINE_AA);
                drawLabel(out, "Document detected - tap capture",
                          Point(12, (int) (40 * scale) + 10), GREEN, scale);
            } else {
                drawLabel(out, "Searching for document...",
                          Point(12, (int) (40 * scale) + 10), YELLOW, scale);
            }
            break;
        }
        default:
            rgba.copyTo(out);
            break;
    }

    auto end = chrono::high_resolution_clock::now();
    return (jlong) chrono::duration_cast<chrono::microseconds>(end - start).count();
}

// Warps the detected document to a flat top-down image. Returns false if no quad.
extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_detectapp_MainActivity_nativeWarpDocument(
        JNIEnv * /*env*/, jobject /*thiz*/,
        jlong inAddr, jlong outAddr, jdouble t1, jdouble t2) {

    Mat &rgba = *reinterpret_cast<Mat *>(inAddr);
    Mat &out = *reinterpret_cast<Mat *>(outAddr);

    vector<Point> quad = findDocQuad(rgba, t1, t2);
    if (quad.size() != 4) return JNI_FALSE;

    Point2f src[4];
    orderPoints(quad, src);

    float widthA = (float) norm(src[2] - src[3]);
    float widthB = (float) norm(src[1] - src[0]);
    float heightA = (float) norm(src[1] - src[2]);
    float heightB = (float) norm(src[0] - src[3]);
    int maxW = (int) max(widthA, widthB);
    int maxH = (int) max(heightA, heightB);
    if (maxW < 10 || maxH < 10) return JNI_FALSE;

    Point2f dst[4] = {{0, 0}, {(float) (maxW - 1), 0},
                      {(float) (maxW - 1), (float) (maxH - 1)}, {0, (float) (maxH - 1)}};
    Mat M = getPerspectiveTransform(src, dst);

    Mat warped;
    warpPerspective(rgba, warped, M, Size(maxW, maxH));
    enhanceForOcr(warped, out);
    return JNI_TRUE;
}

// Enhances a full (un-warped) photo in place for OCR.
extern "C"
JNIEXPORT void JNICALL
Java_com_example_detectapp_MainActivity_nativeEnhance(
        JNIEnv * /*env*/, jobject /*thiz*/, jlong inAddr, jlong outAddr) {
    Mat &in = *reinterpret_cast<Mat *>(inAddr);
    Mat &out = *reinterpret_cast<Mat *>(outAddr);
    enhanceForOcr(in, out);
}
