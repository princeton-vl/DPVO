#include <pybind11/pybind11.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <thread>
#include <tuple>
#include <pybind11/stl.h>
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase
#include <pybind11/numpy.h>

using namespace DBoW2;
namespace py = pybind11;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

void changeStructure(const cv::Mat &plain, std::vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
    out[i] = plain.row(i);
}

void estimateAffine3D(const py::array_t<uint8_t> &pointsA, const py::array_t<uint8_t> &pointsB){

        const py::buffer_info buf = pointsA.request();
        const cv::Mat lpts(buf.shape[0], 3, CV_64F, (double *)buf.ptr);
        std::cout << lpts << std::endl;
        // const cv::Mat image(buf.shape[0], buf.shape[1], CV_64F, (unsigned char*)buf.ptr);

}

typedef std::vector<std::tuple<double, double, double, double, double>> MatchList;

class DPRetrieval {       // The class
  private:
    std::vector<std::vector<cv::Mat > > features;
    std::vector<cv::Mat > descs;
    std::vector<std::vector<cv::KeyPoint > > kps;
    OrbDatabase db;
    const int rad;

  public:             // Access specifier

    DPRetrieval(const std::string vocab_path, const int rad) : rad(rad){

      std::cout << "Loading the vocabulary " << vocab_path << std::endl;

      // load the vocabulary from disk
      OrbVocabulary voc;
      voc.loadFromTextFile(vocab_path);

      db = OrbDatabase(voc, false, 0); // false = do not use direct index
      // (so ignore the last param)
      // The direct index is useful if we want to retrieve the features that
      // belong to some vocabulary node.
      // db creates a copy of the vocabulary, we may get rid of "voc" now

    }

    void insert_image(const py::array_t<uint8_t> &array){

        const py::buffer_info buf = array.request();
        if ((buf.shape.size() != 3) || buf.shape[2] != 3)
          throw std::invalid_argument( "invalid image shape" );

        const cv::Mat image(buf.shape[0], buf.shape[1], CV_8UC3, (unsigned char*)buf.ptr);
        // const cv::Mat image = cv::imread(filepath, 0);

        // cv::imshow("test", image);
        // cv::waitKey(0);

        cv::Mat mask;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        cv::Ptr<cv::ORB> orb = cv::ORB::create();
        orb->detectAndCompute(image, mask, keypoints, descriptors);

        std::vector<cv::Mat > feats;
        changeStructure(descriptors, feats);
        kps.push_back(keypoints);
        features.push_back(feats);
        descs.push_back(descriptors);

        db.add(features.back());

    }

    MatchList match_pair(const int ti, const int qi) const {
      cv::BFMatcher matcher(cv::NORM_HAMMING, true);
      cv::Mat train_descriptors = descs.at(ti);
      cv::Mat query_descriptors = descs.at(qi);

      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher.knnMatch(query_descriptors, train_descriptors, knn_matches, 1);  // Finds the 2 best matches for each descriptor

      MatchList output;


      for (const auto &pair_list : knn_matches){
        if (!pair_list.empty()){
          const auto &pair = pair_list.back();

          auto trainpt = (kps[ti][pair.trainIdx].pt);
          auto querypt = (kps[qi][pair.queryIdx].pt);

          output.emplace_back(trainpt.x, trainpt.y, querypt.x, querypt.y, pair.distance);
        }
      }

      return output;
    }



    auto query(const int i) const {

      if ((i >= features.size()) || (i < 0))
        throw std::invalid_argument( "index invalid" );

      QueryResults ret;
      db.query(features[i], ret, 4);
      std::tuple<float, int, MatchList> output(-1, -1, {});
      for (const auto &r : ret){
        int j = r.Id;
        if ((abs(j - i) >= rad) && (r.Score > std::get<0>(output))){
          const MatchList matches = match_pair(i, j);
          output = std::make_tuple(r.Score, j, matches);
        }
      }
      return output;

    }
};




PYBIND11_MODULE(dpretrieval, m) {

  py::class_<DPRetrieval>(m, "DPRetrieval")
    .def(py::init<std::string, int>())
    .def("insert_image", &DPRetrieval::insert_image)
    .def("match_pair", &DPRetrieval::match_pair)
    .def("query", &DPRetrieval::query);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
