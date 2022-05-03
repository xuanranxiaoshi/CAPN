from setuptools import find_packages, setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 头文件目录
origin_dirs = os.path.dirname(os.path.abspath(__file__))
# 源代码目录
#include = glob.glob(os.path.join(origin_dirs, 'include', '*.h'))
source = glob.glob(os.path.join(origin_dirs, 'src', '*.cpp')) + glob.glob(os.path.join(origin_dirs, 'src', '*.cu'))
#print(include)
print(source )


setup(
    name='pointnet2_ops',  # 模块名称，需要在python中调用
    version="0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=source,
            include_dirs=[os.path.join(origin_dirs,"include")],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    include_package_data=True
)