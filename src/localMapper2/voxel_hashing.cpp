#include <cuda_runtime_api.h>
#include "safe_call.h"
#include "map_struct.h"
#include "map_proc.h"
#include "voxel_hashing.h"

FUSION_HOST void *deviceMalloc(size_t sizeByte)
{
  void *dev_ptr;
  safe_call(cudaMalloc((void **)&dev_ptr, sizeByte));
  return dev_ptr;
}

FUSION_HOST void deviceRelease(void **dev_ptr)
{
  if (*dev_ptr != NULL)
    safe_call(cudaFree(*dev_ptr));

  *dev_ptr = 0;
}

DenseMapping::DenseMapping(int w, int h, const Eigen::Matrix3d &K) : cam_params(K)
{
  device_map.create();
  zrange_x.create(h / 8, w / 8, CV_32FC1);
  zrange_y.create(h / 8, w / 8, CV_32FC1);

  visible_blocks = (HashEntry *)deviceMalloc(sizeof(HashEntry) * device_map.state.num_total_hash_entries_);
  rendering_blocks = (RenderingBlock *)deviceMalloc(sizeof(RenderingBlock) * 100000);

  reset_mapping();
}

DenseMapping::~DenseMapping()
{
  device_map.release();
  deviceRelease((void **)&visible_blocks);
  deviceRelease((void **)&rendering_blocks);
}

void DenseMapping::update(
    cv::cuda::GpuMat depth,
    cv::cuda::GpuMat image,
    const Sophus::SE3d pose)
{
  count_visible_block = 0;

  ::update(
      device_map.map,
      device_map.state,
      depth,
      image,
      pose,
      cam_params,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::fuseFrame(GMat depth, const SE3 &T)
{
  count_visible_block = 0;

  ::update(
      device_map.map,
      device_map.state,
      depth,
      depth,
      T,
      cam_params,
      flag,
      pos_array,
      visible_blocks,
      count_visible_block);
}

void DenseMapping::raytrace(GMat &vertex, const SE3 &T)
{
  if (count_visible_block == 0)
    return;

  ::create_rendering_blocks(
      count_visible_block,
      count_rendering_block,
      visible_blocks,
      zrange_x,
      zrange_y,
      rendering_blocks,
      T,
      cam_params);

  if (count_rendering_block != 0)
  {

    ::raycast(
        device_map.map,
        device_map.state,
        vertex,
        vertex,
        zrange_x,
        zrange_y,
        T,
        cam_params);
  }
}

void DenseMapping::reset()
{
}

void DenseMapping::raycast(
    cv::cuda::GpuMat &vmap,
    cv::cuda::GpuMat &image,
    const Sophus::SE3d pose)
{
  if (count_visible_block == 0)
    return;

  ::create_rendering_blocks(
      count_visible_block,
      count_rendering_block,
      visible_blocks,
      zrange_x,
      zrange_y,
      rendering_blocks,
      pose,
      cam_params);

  if (count_rendering_block != 0)
  {

    ::raycast_with_colour(
        device_map.map,
        device_map.state,
        vmap,
        vmap,
        image,
        zrange_x,
        zrange_y,
        pose,
        cam_params);
  }
}

// void DenseMapping::raycast_check_visibility(
//     cv::cuda::GpuMat &vmap,
//     cv::cuda::GpuMat &image,
//     const Sophus::SE3d pose)
// {
//   ::count_visible_entry(
//       device_map.map,
//       device_map.size,
//       cam_params,
//       pose.inverse(),
//       visible_blocks,
//       count_visible_block);

//   raycast(vmap, image, pose);
// }

void DenseMapping::reset_mapping()
{
  device_map.reset();
}

size_t DenseMapping::fetch_mesh_vertex_only(void *vertex)
{
  uint count_triangle = 0;

  ::create_mesh_vertex_only(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex);

  return (size_t)count_triangle;
}

size_t DenseMapping::fetch_mesh_with_normal(void *vertex, void *normal)
{
  uint count_triangle = 0;

  ::create_mesh_with_normal(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex,
      normal);

  return (size_t)count_triangle;
}

size_t DenseMapping::fetch_mesh_with_colour(void *vertex, void *colour)
{
  uint count_triangle = 0;

  ::create_mesh_with_colour(
      device_map.map,
      device_map.state,
      count_visible_block,
      visible_blocks,
      count_triangle,
      vertex,
      colour);

  return (size_t)count_triangle;
}

void DenseMapping::writeMapToDisk(std::string file_name)
{
  MapStruct<false> host_map;
  host_map.create();
  device_map.download(host_map);
  host_map.writeToDisk(file_name);
  host_map.release();
}

void DenseMapping::readMapFromDisk(std::string file_name)
{
  MapStruct<false> host_map;
  host_map.create();
  host_map.readFromDisk(file_name);
  device_map.upload(host_map);
  host_map.release();
}
