from setuptools import setup

setup(
    name='easyxrd',
    version='0.1.2',
    description='X-ray diffractioan analysis tool',
    url='https://github.com/MehmetTopsakal/easyXRD',
    author='Mehmet Topsakal',
    author_email='metokal@gmail.com',
    license='GNU',
    python_requires='>=3.11',
    packages=['easyxrd'],
    install_requires=['xarray',
                      'numpy',
                      'scipy',
                      'pymatgen',
                      'pyfai',
                      'pybaselines',
                      'ipympl',
                      'GitPython',
                      'mp-api',
                      'fabio',
                      'pandas',
                      ],


    platforms=['any'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.11',
    ],
)
