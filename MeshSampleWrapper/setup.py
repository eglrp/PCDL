from distutils.core import setup, Extension

module1 = Extension('PointSample',
                    sources = ['../python_wrapper.cpp'],
                    include_dirs=['../'],
                    library_dirs=['./',],
                    libraries=['mesh_sample'])

setup (name = 'PointSample',
       version = '1.0',
       description = 'sample point cloud from mesh',
       ext_modules = [module1])