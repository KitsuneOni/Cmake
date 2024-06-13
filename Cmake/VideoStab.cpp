#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

const int SMOOTHING_WINDOW_SIZE = 19; // Size of the Gaussian filter window
const double GAUSSIAN_STD_DEV = 3.0;  // Standard deviation of the Gaussian filter

void drawBorder(Mat& frame, int borderSize, Scalar color) {
    rectangle(frame, Point(0, 0), Point(frame.cols, frame.rows), color, borderSize);
}

int main(int argc, char** argv) {

    //utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    // Open video file or webcam
    VideoCapture cap;
    if (argc == 2) {
        cap.open(argv[1]);
    }
    else {
        cap.open(0);
    }

    if (!cap.isOpened()) {
        cerr << "Error opening video stream or file" << endl;
        return -1;
    }

    cout << "Using Gaussian filter window size: " << SMOOTHING_WINDOW_SIZE << endl;
    cout << "Using Gaussian filter standard deviation: " << GAUSSIAN_STD_DEV << endl;

    Mat prevGray, gray, frame;
    vector<Mat> frameBuffer, HBuffer(SMOOTHING_WINDOW_SIZE);

    // Wait for SMOOTHING_WINDOW_SIZE frames to fill the buffer
    for (int i = 0; i < SMOOTHING_WINDOW_SIZE; ++i) {
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        frameBuffer.push_back(frame.clone());
        if (i > 0) {
            vector<Point2f> prevPts, currPts;
            goodFeaturesToTrack(prevGray, prevPts, 200, 0.01, 30);
            vector<uchar> status;
            vector<float> err;
            calcOpticalFlowPyrLK(prevGray, gray, prevPts, currPts, status, err);
            vector<Point2f> prevPtsFiltered, currPtsFiltered;
            for (size_t j = 0; j < status.size(); ++j) {
                if (status[j]) {
                    prevPtsFiltered.push_back(prevPts[j]);
                    currPtsFiltered.push_back(currPts[j]);
                }
            }
            Mat H = findHomography(prevPtsFiltered, currPtsFiltered, RANSAC);
            if (H.empty()) H = Mat::eye(3, 3, CV_64F);
            HBuffer[i] = (i == 1) ? H.clone() : HBuffer[i - 1] * H;
        }
        else {
            HBuffer[i] = Mat::eye(3, 3, CV_64F);
        }
        prevGray = gray.clone();
    }

    while (true) {
        //std::cout << "Start \n";

        // Process each frame in the buffer
        for (int i = 0; i < frameBuffer.size(); ++i) {
            // Smooth the homography matrices
            Mat smoothedH = Mat::zeros(3, 3, CV_64F);
            double sumWeights = 0;
            for (int j = -SMOOTHING_WINDOW_SIZE / 2; j <= SMOOTHING_WINDOW_SIZE / 2; ++j) {
                int idx = i + j;
                if (idx < 0) idx = 0;
                if (idx >= HBuffer.size()) idx = HBuffer.size() - 1;
                double weight = exp(-0.5 * (j * j) / (GAUSSIAN_STD_DEV * GAUSSIAN_STD_DEV));
                smoothedH += weight * HBuffer[idx];
                sumWeights += weight;
            }
            smoothedH /= sumWeights;

            // Warp the central frame
            Mat stabilizedFrame;
            warpPerspective(frameBuffer[i], stabilizedFrame, smoothedH, frameBuffer[i].size());

            // Draw green border    
            drawBorder(stabilizedFrame, 3, Scalar(0, 255, 0));

            // Display the frame
            imshow("Stabilized Video", stabilizedFrame);

            /*
            MOVE THIS FROM IN HERE TO OUTSIDE
            // Wait for a key press
            if (waitKey(30) >= 0)
                break;
            */
        }

        //MOVE THAT TO HERE 
         
        // Wait for a key press
        if (waitKey(30) >= 0)
            break;
        
        // Read new frames to keep the buffer full
        cap >> frame;
        if (frame.empty()) break;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        frameBuffer.push_back(frame.clone());
        frameBuffer.erase(frameBuffer.begin());

        vector<Point2f> prevPts, currPts;
        goodFeaturesToTrack(prevGray, prevPts, 200, 0.01, 30);
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(prevGray, gray, prevPts, currPts, status, err);
        vector<Point2f> prevPtsFiltered, currPtsFiltered;
        for (size_t j = 0; j < status.size(); ++j) {
            if (status[j]) {
                prevPtsFiltered.push_back(prevPts[j]);
                currPtsFiltered.push_back(currPts[j]);
            }
        }
        Mat H = findHomography(prevPtsFiltered, currPtsFiltered, RANSAC);
        if (H.empty()) H = Mat::eye(3, 3, CV_64F);
        HBuffer.push_back(HBuffer.back() * H);
        HBuffer.erase(HBuffer.begin());

        prevGray = gray.clone();

        //std::cout << "Finished Loop";
    }

    //Comment these out in order to make the program close on one key press
    //Setup a variable before destroy windows to stop the program from closing
    string inp;

    //input anything and press enter in order to make this work
    cin >> inp;
    

    cap.release();
    destroyAllWindows();
    return 0;
}
