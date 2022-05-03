#include "ball_query.h"
#include "utils.h"

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, int *idx);

void query_ball_point_kernel_wrapper2(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const int *fps_idx,int *idx);

void query_ball_point_kernel_wrapper3(int b, int n, int c, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const int *fps_idx,int *idx);

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);

  if (new_xyz.is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.is_cuda()) {
    query_ball_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data_ptr<float>(),
                                    xyz.data_ptr<float>(), idx.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return idx;
}
at::Tensor ball_query2(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample,at::Tensor fps_idx) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_INT(fps_idx);

  if (new_xyz.is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.is_cuda()) {
    query_ball_point_kernel_wrapper2(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data_ptr<float>(),
                                    xyz.data_ptr<float>(), fps_idx.data_ptr<int>(),idx.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return idx;
}

at::Tensor ball_query3(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample,at::Tensor fps_idx) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_INT(fps_idx);

  if (new_xyz.is_cuda()) {
    CHECK_CUDA(xyz);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.is_cuda()) {
    query_ball_point_kernel_wrapper3(xyz.size(0), xyz.size(1), xyz.size(2), new_xyz.size(1),
                                    radius, nsample, new_xyz.data_ptr<float>(),
                                    xyz.data_ptr<float>(), fps_idx.data_ptr<int>(),idx.data_ptr<int>());
  } else {
    AT_ASSERT(false, "CPU not supported");
  }

  return idx;
}
