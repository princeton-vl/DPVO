#include "viewer_cuda.h"

#define THREADS 64

#define NUM_BLOCKS(batch_size) ((batch_size + THREADS - 1) / THREADS)

#define GPU_1D_KERNEL_LOOP(k, n) \
  for (size_t k = threadIdx.x; k<n; k += blockDim.x)


__device__ void
actSO3(const float* __restrict__ q, 
       const float* __restrict__ X, 
       float* __restrict__ Y) {
  
  float uv[3];
  uv[0] = 2.0 * (q[1]*X[2] - q[2]*X[1]);
  uv[1] = 2.0 * (q[2]*X[0] - q[0]*X[2]);
  uv[2] = 2.0 * (q[0]*X[1] - q[1]*X[0]);

  Y[0] = X[0] + q[3]*uv[0] + (q[1]*uv[2] - q[2]*uv[1]);
  Y[1] = X[1] + q[3]*uv[1] + (q[2]*uv[0] - q[0]*uv[2]);
  Y[2] = X[2] + q[3]*uv[2] + (q[0]*uv[1] - q[1]*uv[0]);
}

__device__  void
actSE3(const float* __restrict__ t, 
       const float* __restrict__ q, 
       const float* __restrict__ X, 
       float* __restrict__ Y) {
  
  actSO3(q, X, Y);
  Y[3] = X[3];
  Y[0] += X[3] * t[0];
  Y[1] += X[3] * t[1];
  Y[2] += X[3] * t[2];
}

__device__ void
invSE3(const float* __restrict__ t, 
       const float* __restrict__ q, 
       float* __restrict__ tinv, 
       float* __restrict__ qinv) {
  qinv[0] = -q[0];
  qinv[1] = -q[1];
  qinv[2] = -q[2];
  qinv[3] =  q[3];
  
  actSO3(qinv, t, tinv);
  tinv[0] = -tinv[0];
  tinv[1] = -tinv[1];
  tinv[2] = -tinv[2];
}


__global__ void iproj_kernel(const int index, const int nFrames, const float thresh,
    const torch::PackedTensorAccessor32<unsigned char,4,torch::RestrictPtrTraits> images,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> disps,
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> intrinsics,
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> points,
    torch::PackedTensorAccessor32<unsigned char,2,torch::RestrictPtrTraits> colors,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> count)
{

  __shared__ float t[3], t1[3], t2[3];
  __shared__ float q[4], q1[4], q2[4];
  __shared__ float intrinsic[4], intrinsic1[4];

  if (threadIdx.x < 3) {
    t[threadIdx.x] = poses[index][threadIdx.x + 0];
  }

  if (threadIdx.x < 4) {
    q[threadIdx.x] = poses[index][threadIdx.x + 3];
  }

  if (threadIdx.x < 4) {
    intrinsic[threadIdx.x] = 8 * intrinsics[index][threadIdx.x];
  }

  __syncthreads();


  if (threadIdx.x == 0) {
    invSE3(t, q, t1, q1);
  }

  __syncthreads();

  const int ht = disps.size(1);
  const int wd = disps.size(2);

  const int k = blockIdx.x * THREADS + threadIdx.x;

  if (k < ht * wd) {
    float X0[4], X1[4], X2[4];
    const int i = k / wd;
    const int j = k % wd;

    count[k] = 0;

    if ((i < ht - 1) && (j < wd - 1)) {
      const float d = disps[index][i][j];
      const float dx = disps[index][i][j+1] - disps[index][i][j];
      const float dy = disps[index][i+1][j] - disps[index][i][j];

      if (sqrt(dx*dx + dy*dy) > 0.01) {
        count[k] = -100;
      }

      X0[0] = ((float) j - intrinsic[2]) / intrinsic[0];
      X0[1] = ((float) i - intrinsic[3]) / intrinsic[1];
      X0[2] = 1;
      X0[3] = d;

      actSE3(t1, q1, X0, X1);

      points[k][0] = X0[0] / X0[3];
      points[k][1] = X0[1] / X0[3];
      points[k][2] = X0[2] / X0[3];

      colors[k][0] = images[index][2][i][j];
      colors[k][1] = images[index][1][i][j];
      colors[k][2] = images[index][0][i][j];


      for (int jx=0; jx < nFrames; jx++) {
        if (jx == index) continue;

        if (threadIdx.x < 3) {
          t2[threadIdx.x] = poses[jx][threadIdx.x + 0];
        }

        if (threadIdx.x < 4) {
          q2[threadIdx.x] = poses[jx][threadIdx.x + 3];
        }

        if (threadIdx.x < 4) {
          intrinsic1[threadIdx.x] = 8 * intrinsics[jx][threadIdx.x];
        }

        __syncthreads();

        actSE3(t2, q2, X1, X2);

        const float x1 = intrinsic1[0] * (X2[0] / X2[2]) + intrinsic1[2];
        const float y1 = intrinsic1[1] * (X2[1] / X2[2]) + intrinsic1[3];

        const int i1 = static_cast<int>(round(y1));
        const int j1 = static_cast<int>(round(x1));

        if ((i1 >= 0) && (i1 < ht) && (j1 >= 0) && (j1 < wd) && (d > 0.1)) {
          const float z1 = disps[jx][i1][j1];
          const float z2 = X2[3] / X2[2];
          
          if (100 * (max(z1/z2, z2/z1) - 1) < thresh) {
            count[k] += 1;
          }
        }
      }
    }
  }
}


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
    const torch::Tensor intrinsics) 
{
  const int ht = disps.size(1);
  const int wd = disps.size(2);

  const int nPoints = ht * wd;
  torch::Tensor points = torch::zeros({nPoints, 3}, disps.options());
  torch::Tensor colors = torch::zeros({nPoints, 3}, images.options());
  torch::Tensor count = torch::zeros({nPoints}, disps.options());

  iproj_kernel<<<NUM_BLOCKS(ht * wd), THREADS>>>(index, nFrames, thresh,
    images.packed_accessor32<unsigned char,4,torch::RestrictPtrTraits>(),
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    disps.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
    intrinsics.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    points.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    colors.packed_accessor32<unsigned char,2,torch::RestrictPtrTraits>(),
    count.packed_accessor32<float,1,torch::RestrictPtrTraits>());

  torch::Tensor m = masks[index].reshape({-1});
  torch::Tensor pointsFiltered, colorsFiltered;

  // std::cout << index << " " << dynamic << std::endl;

  pointsFiltered = torch::zeros({0, 3}, points.options());
  colorsFiltered = torch::zeros({0, 3}, colors.options());

  if (showForeground) {
    pointsFiltered = torch::cat({pointsFiltered, at::index(points, {(count >= 0) & (m < 0.5)})}, 0);
    colorsFiltered = torch::cat({colorsFiltered, at::index(colors, {(count >= 0) & (m < 0.5)})}, 0);
  }

  if (showBackground) {
    pointsFiltered = torch::cat({pointsFiltered, at::index(points, {(count >= 2.0) & (m > 0.5)})}, 0);
    colorsFiltered = torch::cat({colorsFiltered, at::index(colors, {(count >= 2.0) & (m > 0.5)})}, 0);
  }


  const int mPoints = pointsFiltered.size(0);

  return {mPoints, pointsFiltered, colorsFiltered};
}


__global__ void pose_to_matrix_kernel(
    const torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> poses,
    torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> mat4x4)
{

  const int index = blockIdx.x * THREADS + threadIdx.x;

  float t0[3], t[3];
  float q0[4], q[4];

  if (index < poses.size(0)) {

    t0[0] = poses[index][0];
    t0[1] = poses[index][1];
    t0[2] = poses[index][2];

    q0[0] = poses[index][3];
    q0[1] = poses[index][4];
    q0[2] = poses[index][5];
    q0[3] = poses[index][6];

    invSE3(t0, q0, t, q);

    mat4x4[index][0][0] = 1 - 2*q[1]*q[1] - 2*q[2]*q[2];
    mat4x4[index][0][1] = 2*q[0]*q[1] - 2*q[3]*q[2];
    mat4x4[index][0][2] = 2*q[0]*q[2] + 2*q[3]*q[1];
    mat4x4[index][0][3] = t[0];

    mat4x4[index][1][0] = 2*q[0]*q[1] + 2*q[3]*q[2];
    mat4x4[index][1][1] = 1 - 2*q[0]*q[0] - 2*q[2]*q[2];
    mat4x4[index][1][2] = 2*q[1]*q[2] - 2*q[3]*q[0];
    mat4x4[index][1][3] = t[1];

    mat4x4[index][2][0] = 2*q[0]*q[2] - 2*q[3]*q[1]; 
    mat4x4[index][2][1] = 2*q[1]*q[2] + 2*q[3]*q[0]; 
    mat4x4[index][2][2] = 1 - 2*q[0]*q[0] - 2*q[1]*q[1];
    mat4x4[index][2][3] = t[2];

    mat4x4[index][3][0] = 0.0;
    mat4x4[index][3][1] = 0.0;
    mat4x4[index][3][2] = 0.0;
    mat4x4[index][3][3] = 1.0;

  }
}


torch::Tensor poseToMatrix(const torch::Tensor poses) {
  const int num = poses.size(0);
  torch::Tensor mat4x4 = torch::zeros({num, 4, 4}, poses.options());

  pose_to_matrix_kernel<<<NUM_BLOCKS(num), THREADS>>>(
    poses.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
    mat4x4.packed_accessor32<float,3,torch::RestrictPtrTraits>());

  return mat4x4;
}