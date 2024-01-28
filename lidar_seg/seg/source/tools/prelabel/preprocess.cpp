#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>
#include <cfloat>

#include <iostream>

torch::Tensor build_feature_map(torch::Tensor cloud,
  int cs, int hs, int ws, float min_x, float min_y, float min_z, float max_z,
  float xy_res, float z_res) {
  torch::Tensor fm_at = torch::zeros({cs, hs, ws}, torch::kFloat32);  // CHW.
  auto fm_at_scalar = fm_at.accessor<float, 3>();

  const int non_z_cha_num = cs - (max_z - min_z) / z_res;
  auto point_num = cloud.sizes()[0];
  auto cloud_scalar = cloud.accessor<float, 2>();
  for (int i = 0; i < point_num; ++i) {
    if (std::isnan(cloud_scalar[i][0]) || std::isnan(cloud_scalar[i][1]) || std::isnan(cloud_scalar[i][2])) {
      continue;
    }

    const int idx_h = static_cast<int>(std::floor((cloud_scalar[i][0] - min_x) / xy_res));
    const int idx_w = static_cast<int>(std::floor((cloud_scalar[i][1] - min_y) / xy_res));
    int idx_c = static_cast<int>(std::floor((cloud_scalar[i][2] - min_z) / z_res)) + non_z_cha_num;
    idx_c = std::max(non_z_cha_num, idx_c);
    idx_c = std::min(cs - 1, idx_c);
    if (idx_h >= hs || idx_h < 0 || idx_w >= ws || idx_w < 0) {
      continue;
    }

    fm_at_scalar[idx_c][idx_h][idx_w] += 1.0F;
    fm_at_scalar[0][idx_h][idx_w] = (idx_h + 0.5F) * xy_res + min_x;
    fm_at_scalar[1][idx_h][idx_w] = (idx_w + 0.5F) * xy_res + min_y;
  }

  return fm_at;
}

torch::Tensor augment_cloud(torch::Tensor cloud,
  const float sin_a, const float cos_a,
  const float dx, const float dy, const float dz) {
  auto point_num = cloud.sizes()[0];
  auto cloud_scalar = cloud.accessor<float, 2>();
  for (int i = 0; i < point_num; ++i) {
    if (std::isnan(cloud_scalar[i][0]) || std::isnan(cloud_scalar[i][1]) || std::isnan(cloud_scalar[i][2])) {
      continue;
    }

    const float x_new = cloud_scalar[i][0] * cos_a - cloud_scalar[i][1] * sin_a + dx;
    const float y_new = cloud_scalar[i][0] * sin_a + cloud_scalar[i][1] * cos_a + dy;
    const float z_new = cloud_scalar[i][2] + dz;
    cloud_scalar[i][0] = x_new;
    cloud_scalar[i][1] = y_new;
    cloud_scalar[i][2] = z_new;
  }

  return cloud;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build", &build_feature_map, "build feature map as input.");
  m.def("augment", &augment_cloud, "augment cloud for training.");
}