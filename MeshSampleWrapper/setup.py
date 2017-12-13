from distutils.core import setup, Extension

module1 = Extension('point_sample',
                    sources = ['python_wrapper.cpp'],
                    include_dirs=['./'],
                    library_dirs=['./cmake-build-release',],
                    libraries=['mesh_sample'])

setup (name = 'point_sample',
       version = '1.0',
       description = 'sample point cloud from mesh',
       ext_modules = [module1])