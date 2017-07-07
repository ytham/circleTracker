#include <assert.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "constants.h"
#include <queue>
#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace std;

void detectAndOutlineEyes(Mat frame);
Point findEyeCenter(Mat input, Rect eye, string debugWindow);

string cascade_type = "haarcascade_eye.xml";
CascadeClassifier eye_cascade;
int mFrameNumber = 0;

int main(int argc, char** argv) {
  // VideoCapture::VideoCapture capture(0);
  VideoCapture::VideoCapture capture("eyeVideos/09073E216F5DF458.MOV");
  if (!capture.isOpened()) {
    return -1;
  }

  // // Process image
  // Mat src, src_gray, src_thresh, dst;
  // src = imread("eye.png", 1);
  // if (!src.data) {
  //   return -1;
  // }
  // cvtColor(src, src_gray, CV_BGR2GRAY);

  // Load cascade classifier
  if (!eye_cascade.load(cascade_type)) {
    printf("Error loading eye cascade");
    return -1;
  }

  // Read camera video stream
  Mat frame;
  while (true) {
    // Only capture every 30th frame
    if (mFrameNumber % 30 != 0) {
      mFrameNumber++;
      continue;
    }
    capture >> frame;

    // Apply classifier and display frame
    if (!frame.empty()) {
      detectAndOutlineEyes(frame);
    } else {
      printf("No frame data captured");
      break;
    }
    mFrameNumber++;
    waitKey(0);
  }

  // GaussianBlur(src_gray, src_gray, Size(9,9),2,2);
  // adaptiveThreshold(src_gray, src_thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 9, 0);

  // vector<Vec3f> circles;
  //
  // HoughCircles(src_thresh, circles, CV_HOUGH_GRADIENT, 3, src_gray.rows/8, 200, 100, 20, 150);
  //
  // cvtColor(src_thresh, dst, CV_GRAY2RGB);
  //
  // for (size_t i = 0; i < circles.size(); i++) {
  //   Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
  //   int radius = cvRound(circles[i][2]);
  //   circle(dst, center, 3, Scalar(0, 255, 0), -1, 8, 0);
  //   circle(dst, center, radius, Scalar(0, 0, 255), 3, 8, 0);
  // }
  //
  // resize(dst, dst, Size(), 0.5, 0.5);
  //
  // cout << circles.size() << endl;
  //
  //
  // namedWindow("HoughCircles", CV_WINDOW_AUTOSIZE);
  // imshow("HoughCircles", dst);
  //
  // waitKey(0);
  return 0;
}

void scaleToFastSize(const Mat &src,Mat &dst) {
  resize(src, dst, Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

void detectAndOutlineEyes(Mat frame) {
  // Prepare image
  Mat frame_gray;
  cvtColor(frame, frame_gray, CV_BGR2GRAY);

  Size resized_size(frame_gray.cols/4, frame_gray.rows/4);
  resize(frame_gray, frame_gray, resized_size);

  transpose(frame_gray, frame_gray);
  flip(frame_gray, frame_gray, 0);

  Rect crop = Rect(70, 70, 190, 150);

  Mat frame_gray_crop = frame_gray(crop);

  // Apply CV - Traditional functions
  Point eyeCenter = findEyeCenter(frame_gray_crop, Rect(0,0,frame_gray_crop.cols,frame_gray_crop.rows), "Debug");

  circle(frame_gray_crop, eyeCenter, 10, Scalar(255,0,0), 4, 8, 0);

  // // Apply CV - HAAR Cascade
  // vector<Rect> eyes;
  // eye_cascade.detectMultiScale(frame_gray, eyes, 1.3, 5, 0, Size(20,20), Size(500,500));
  //
  // for (size_t i = 0; i < eyes.size(); i++) {
  //   Point center(eyes[i].x+(eyes[i].width/2), eyes[i].y+(eyes[i].height/2));
  //   int radius = cvRound((eyes[i].width + eyes[i].height)/4);
  //   circle(frame_gray, center, 10, Scalar(255,0,0), 4, 8, 0);
  //   if (i == 0) {
  //     cout << "XY: " << center.x << "," << center.y << endl;
  //   }
  // }

  imshow("Eye Tracking", frame_gray_crop);
}

bool inMat(Point p,int rows,int cols) {
  return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

Mat matrixMagnitude(const Mat &matX, const Mat &matY) {
  Mat mags(matX.rows,matX.cols,CV_64F);
  for (int y = 0; y < matX.rows; ++y) {
    const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
    double *Mr = mags.ptr<double>(y);
    for (int x = 0; x < matX.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = sqrt((gX * gX) + (gY * gY));
      Mr[x] = magnitude;
    }
  }
  return mags;
}

double computeDynamicThreshold(const Mat &mat, double stdDevFactor) {
  Scalar stdMagnGrad, meanMagnGrad;
  meanStdDev(mat, meanMagnGrad, stdMagnGrad);
  double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
  return stdDevFactor * stdDev + meanMagnGrad[0];
}

Mat computeMatXGradient(const Mat &mat) {
  Mat out(mat.rows,mat.cols,CV_64F);

  for (int y = 0; y < mat.rows; ++y) {
    const uchar *Mr = mat.ptr<uchar>(y);
    double *Or = out.ptr<double>(y);

    Or[0] = Mr[1] - Mr[0];
    for (int x = 1; x < mat.cols - 1; ++x) {
      Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
    }
    Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
  }

  return out;
}

void testPossibleCentersFormula(int x, int y, const Mat &weight,double gx, double gy, Mat &out) {
  // for all possible centers
  for (int cy = 0; cy < out.rows; ++cy) {
    double *Or = out.ptr<double>(cy);
    const unsigned char *Wr = weight.ptr<unsigned char>(cy);
    for (int cx = 0; cx < out.cols; ++cx) {
      if (x == cx && y == cy) {
        continue;
      }
      // create a vector from the possible center to the gradient origin
      double dx = x - cx;
      double dy = y - cy;
      // normalize d
      double magnitude = sqrt((dx * dx) + (dy * dy));
      dx = dx / magnitude;
      dy = dy / magnitude;
      double dotProduct = dx*gx + dy*gy;
      dotProduct = max(0.0,dotProduct);
      // square and multiply by the weight
      if (kEnableWeight) {
        Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
      } else {
        Or[cx] += dotProduct * dotProduct;
      }
    }
  }
}

bool floodShouldPushPoint(const Point &np, const Mat &mat) {
  return inMat(np, mat.rows, mat.cols);
}

// returns a mask
Mat floodKillEdges(Mat &mat) {
  rectangle(mat,Rect(0,0,mat.cols,mat.rows),255);

  Mat mask(mat.rows, mat.cols, CV_8U, 255);
  queue<Point> toDo;
  toDo.push(Point(0,0));
  while (!toDo.empty()) {
    Point p = toDo.front();
    toDo.pop();
    if (mat.at<float>(p) == 0.0f) {
      continue;
    }
    // add in every direction
    Point np(p.x + 1, p.y); // right
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x - 1; np.y = p.y; // left
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y + 1; // down
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y - 1; // up
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    // kill it
    mat.at<float>(p) = 0.0f;
    mask.at<uchar>(p) = 0;
  }
  return mask;
}

Point unscalePoint(Point p, Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return Point(x,y);
}

Point findEyeCenter(Mat input, Rect eye, string debugWindow) {
  Mat eyeROIUnscaled = input(eye);
  Mat eyeROI;
  scaleToFastSize(eyeROIUnscaled, eyeROI);
  // Mat eyeROI = input(eye);
  // draw eye region
  rectangle(input,eye,1234);
  //-- Find the gradient
  Mat gradientX = computeMatXGradient(eyeROI);
  Mat gradientY = computeMatXGradient(eyeROI.t()).t();
  //-- Normalize and threshold the gradient
  // compute all the magnitudes
  Mat mags = matrixMagnitude(gradientX, gradientY);
  //compute the threshold
  double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
  //double gradientThresh = kGradientThreshold;
  //double gradientThresh = 0;
  //normalize
  for (int y = 0; y < eyeROI.rows; ++y) {
    double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    const double *Mr = mags.ptr<double>(y);
    for (int x = 0; x < eyeROI.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = Mr[x];
      if (magnitude > gradientThresh) {
        Xr[x] = gX/magnitude;
        Yr[x] = gY/magnitude;
      } else {
        Xr[x] = 0.0;
        Yr[x] = 0.0;
      }
    }
  }
  // imshow(debugWindow,gradientX);
  //-- Create a blurred and inverted image for weighting
  Mat weight;
  GaussianBlur( eyeROI, weight, Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
  for (int y = 0; y < weight.rows; ++y) {
    unsigned char *row = weight.ptr<unsigned char>(y);
    for (int x = 0; x < weight.cols; ++x) {
      row[x] = (255 - row[x]);
    }
  }
  // imshow(debugWindow,weight);
  //-- Run the algorithm!
  Mat outSum = Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
  // for each possible gradient location
  // Note: these loops are reversed from the way the paper does them
  // it evaluates every possible center for each gradient location instead of
  // every possible gradient location for every center.
  printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);
  for (int y = 0; y < weight.rows; ++y) {
    const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    for (int x = 0; x < weight.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      if (gX == 0.0 && gY == 0.0) {
        continue;
      }
      testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
    }
  }
  // scale all the values down, basically averaging them
  double numGradients = (weight.rows*weight.cols);
  Mat out;
  outSum.convertTo(out, CV_32F,1.0/numGradients);
  // imshow(debugWindow,out);
  //-- Find the maximum point
  Point maxP;
  double maxVal;
  minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
  //-- Flood fill the edges
  if(kEnablePostProcess) {
    Mat floodClone;
    //double floodThresh = computeDynamicThreshold(out, 1.5);
    double floodThresh = maxVal * kPostProcessThreshold;
    threshold(out, floodClone, floodThresh, 0.0f, THRESH_TOZERO);
    if(kPlotVectorField) {
      //plotVecField(gradientX, gradientY, floodClone);
      // imwrite("eyeFrame.png",eyeROIUnscaled);
    }
    Mat mask = floodKillEdges(floodClone);
    //imshow(debugWindow + " Mask",mask);
    imshow(debugWindow,out);
    // redo max
    minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
  }
  return unscalePoint(maxP,eye);
}
