#include <torch/extension.h>


struct PointCloud {
    const int nPoints;
    const torch::Tensor points;
    const torch::Tensor colors;
};

PointCloud backproject_and_filter(
    const int index,
    const int nFrames,
    const float thresh,
    const bool showForeground,
    const bool showBackground,
    const torch::Tensor images,
    const torch::Tensor poses,
    const torch::Tensor disps,
    const torch::Tensor masks,
    const torch::Tensor intrinsics);


torch::Tensor poseToMatrix(torch::Tensor poses);