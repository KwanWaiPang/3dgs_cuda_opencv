# 3DGS的cpp实现

* [3D-GS的源码](https://github.com/graphdeco-inria/gaussian-splatting)
* [注释版本的3DGS](https://github.com/arclab-hku/comment_3DGS)
* [此仓库在笔记本Ubuntu20.04下的编译过程记录](https://blog.csdn.net/gwplovekimi/article/details/136348402?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22136348402%22%2C%22source%22%3A%22gwplovekimi%22%7D)
* [c++以及CUDA实现，gaussian-splatting-cuda](https://github.com/MrNeRF/gaussian-splatting-cuda)目前star最多的版本
* [纯c++实现，OpenSplat](https://github.com/pierotofy/OpenSplat)
* [C++以及OpenGL](https://github.com/hyperlogic/splatapult)
* [采用Vulkan API](https://github.com/shg8/VulkanSplatting)

## gaussian-splatting-cuda
* [Photo-SLAM](https://arxiv.org/pdf/2311.16728.pdf)也是采用LibTorch;

### Build
* 由于不用pytorch，所以也不用conda了~
* CMake 3.24 or higher is required. 需要更新CMakelist中的版本
~~~
#查看CMake的版本号（要变更）：
cmake --version

#查看cuda版本
nvcc --version
~~~

* 配置过程：
~~~
cd /home/gwp/LiDAR-3DGS/gaussian-splatting-cuda-master

# 下载并解压libtorch，这是pytorch的cpp版本
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip  

unzip  libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip -d external/

#移除掉（可选）
rm libtorch-cxx11-abi-shared-with-deps-2.0.1+cu118.zip

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j16

#数据集下载
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip

unzip tandt_db.zip -d data_set/

rm tandt_db.zip

~~~

### 运行
~~~
./build/gaussian_splatting_cuda -d ~/dataset/tandt_db/tandt/truck -o ~/catkin_ws/src/LiDAR-3DGS/gaussian-splatting-cuda-master/3DGS_result -i 6000
~~~

### 可视化
~~~
git clone https://github.com/camenduru/sibr_core.git
git checkout fossa_compatibility
cd SIBR_viewers
cmake -Bbuild .
cmake --build build -j24 --target install
#cmake --build build --target install --config RelWithDebInfo

~~~
结果可视化
~~~
cd SIBR_viewers
source embree-3.6.1.x86_64.linux/embree-vars.sh
cd ..
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m ~/catkin_ws/src/LiDAR-3DGS/gaussian-splatting-cuda-master/3DGS_result
~~~


## OpenSplat
* for cpu-only
* 运行的方式是`./opensplat /path/to/banana -n 2000`，也就是`opensplat`为启动文件~


* 获取新的分支到本地
```
git fetch origin 分支名
```
