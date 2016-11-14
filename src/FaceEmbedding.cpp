/*
* Based on SeetaFaceEngine/FaceIdentification/src/test/test_face_verification.cpp
*
* This file is part of the open-source SeetaFace engine, which includes three modules:
* SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
*
* This file is part of the SeetaFace Identification module, containing codes implementing the
* face identification method described in the following paper:
*
*
*   VIPLFaceNet: An Open Source Deep Face Recognition SDK,
*   Xin Liu, Meina Kan, Wanglong Wu, Shiguang Shan, Xilin Chen.
*   In Frontiers of Computer Science.
*
*
* Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
* Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
*
* The codes are mainly developed by Jie Zhang(a Ph.D supervised by Prof. Shiguang Shan)
*
* As an open-source face recognition engine: you can redistribute SeetaFace source codes
* and/or modify it under the terms of the BSD 2-Clause License.
*
* You should have received a copy of the BSD 2-Clause License along with the software.
* If not, see < https://opensource.org/licenses/BSD-2-Clause>.
*
* Contact Info: you can send an email to SeetaFace@vipl.ict.ac.cn for any problems.
*
* Note: the above information must be kept whenever or wherever the codes are used.
*
*/

#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

#if defined(__unix__) || defined(__APPLE__)

#ifndef fopen_s

#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL

#endif //fopen_s

#endif //__unix

#include <opencv2/opencv.hpp>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

using namespace seeta;

#ifdef _WIN32
std::string FD_MODEL = "model/seeta_fd_frontal_v1.0.bin";
std::string FA_MODEL = "model/seeta_fa_v1.1.bin";
std::string FR_MODEL = "model/seeta_fr_v1.0.bin";
#else
std::string FD_MODEL = "../SeetaFaceEngine/FaceDetection/model/seeta_fd_frontal_v1.0.bin";
std::string FA_MODEL = "../SeetaFaceEngine/FaceAlignment/model/seeta_fa_v1.1.bin";
std::string FR_MODEL = "../SeetaFaceEngine/FaceIdentification/model/seeta_fr_v1.0.bin";
#endif

void usage() {
  std::cout << "Usage: FaceEmbedding list_file.txt output_embedding_file.csv" << std::endl
    << "list_file.txt:" << std::endl
    << "path/to/image1.jpg" << std::endl
    << "path/to/image2.jpg" << std::endl
    << "..." << std::endl;
}

int main(int argc, char* argv[]) {
  std::string listName;  
  std::string feaName;
  
  std::string fdModel = FD_MODEL;
  std::string faModel = FA_MODEL;
  std::string frModel = FR_MODEL;
  
  usage();
    
  if (argc > 1) listName = argv[1];
  if (argc > 2) feaName = argv[2];
    
  // Initialize face detection model
  seeta::FaceDetection detector(fdModel.c_str());
  detector.SetMinFaceSize(40);
  detector.SetScoreThresh(2.f);
  detector.SetImagePyramidScaleFactor(0.8f);
  detector.SetWindowStep(4, 4);

  // Initialize face alignment model 
  seeta::FaceAlignment point_detector(faModel.c_str());

  // Initialize face Identification model 
  FaceIdentification face_recognizer(frModel.c_str());
  
  std::ofstream ofs;
  if (feaName.size() > 0) {
    ofs.open(feaName, ios_base::out | ios_base::trunc);
    ofs << std::setprecision(18);
  }
    
  std::ifstream ifs(listName);
  std::string imageName;
  int total = 0;
  int notDetected = 0;
  while (1) {
    getline(ifs, imageName);
    if (!ifs.good()) break;
    
    if (1) {
      // ignore first column (image id) if image list file is in csv format
      size_t pos = imageName.find_first_of(',');
      if (pos != std::string::npos) {
        imageName.erase(0, pos + 1);
      }
    }
    
    cv::Mat img_color = cv::imread(imageName, 1);
    if (img_color.empty()) {
      std::cout << "Unable to read " << imageName << std::endl;
      continue;
    }
    cv::Mat img_gray;
    cv::cvtColor(img_color, img_gray, cv::COLOR_BGR2GRAY);
    
    ImageData img_data_color(img_color.cols, img_color.rows, img_color.channels());
    img_data_color.data = img_color.data;    
    
    ImageData img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
    img_data_gray.data = img_gray.data;    

    // Detect faces
    std::vector<seeta::FaceInfo> faces = detector.Detect(img_data_gray);
    int32_t face_num = static_cast<int32_t>(faces.size());
    
    cv::Mat crop(face_recognizer.crop_height(), face_recognizer.crop_width(), CV_8UC3);
    ImageData crop_data(crop.cols, crop.rows, crop.channels());
    crop_data.data = crop.data;    
    
    std::cout << imageName << " ";
    
    bool faceDetected = true;
    
    if (face_num <= 0) {
      std::cout << "unable to detect face ";
      // set face bounding box to the whole image
      seeta::FaceInfo info;
      memset(&info, 0, sizeof(info));
      info.bbox.x = 0;
      info.bbox.y = 0;
      info.bbox.width = img_gray.cols;
      info.bbox.height = img_gray.rows;      
      faces.push_back(info);

      faceDetected = false;
      notDetected++;
    }
    
    if (faces.size() > 0) {
      // Detect 5 facial landmarks
      seeta::FacialLandmark points[5];
      bool detected = point_detector.PointDetectLandmarks(img_data_gray, faces[0], points);
      
      if (!detected) {
        std::cout << "unable to detect facial landmarks ";
        // assume that face image is already cropped
        cv::resize(img_color, crop, crop.size());
      }
      else {
        face_recognizer.CropFace(img_data_color, points, crop_data);
      }
      
      if (0 && !faceDetected) {
        cv::Mat view = img_color.clone();
        for (int i = 0; i < 5; i++) {
          cv::circle(view, cv::Point((int) points[i].x, (int)points[i].y), 3,
			  cv::Scalar(255, 0, 0), 1);
        }
        cv::imshow("points", view);
        cv::imshow("crop", crop);
        int key = cv::waitKey(0);
        if (key == 27) break;
      }
    }
    
    float fea[2048];
    memset(fea, 0, sizeof(fea));
    int feaSize = face_recognizer.feature_size();
    
    face_recognizer.ExtractFeature(crop_data, fea);
    
	float norm = 0;
	for (int i = 0; i < feaSize; i++) {
		norm += fea[i] * fea[i];
	}
	norm = sqrt(norm);
    if (norm > 0) {
      float s = 1.f/norm;
      for (int i = 0; i < feaSize; i++) fea[i] *= s;
    }
    
    if (ofs.is_open()) {
      int i = 0;
      for (i = 0; i < feaSize-1; i++) {
        ofs << fea[i] << ",";
      }
      ofs << fea[i] << std::endl;
    }
    
    std::cout << std::endl;
    total++;
    
    if (0) {
      cv::imshow("src", img_color);
      cv::imshow("crop", crop);
      int key = cv::waitKey(0);
      if (key == 27) break;
    }
  }
  
  std::cout << "Total images: " << total << endl
    << "Faces not detected: " << notDetected << endl;

  return 0;
}
