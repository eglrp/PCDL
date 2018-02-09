from distutils.core import setup, Extension

module1 = Extension('PointsUtil',
                    sources = ['./points_util_py.cpp'],
                    include_dirs=['.'],
                    library_dirs=['.','build'],
                    libraries=['points2voxel'])

setup (name = 'PointsUtil',
       version = '1.0',
       description = '',
       ext_modules = [module1])