#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  Mat src;//, src_gray, src_thresh, dst;

  // Load the image
  src = imread("eye.png", 1);

  if (!src.data) {
    return -1;
  }

  // Downsample

  // Detect face
  Mat src_gray, faces, detected;
  cvtColor(src, src_gray, CV_BGR2GRAY);
  cascadeClassifier("haarcascade_eye.xml", faces);
  faces.detectMultiScale(frame, detected, 1.3, 5); // ??

  Mat pupilFrame, pupil0;


  // Draw square
  for ()
}
