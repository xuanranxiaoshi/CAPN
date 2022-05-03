#pragma once
#include <torch/extension.h>

at::Tensor ball_query(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample);
at::Tensor ball_query2(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample,at::Tensor fps_idx);
at::Tensor ball_query3(at::Tensor new_xyz, at::Tensor xyz, const float radius,
                      const int nsample,at::Tensor fps_idx);
