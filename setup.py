from setuptools import setup

setup(
  name='tinypfn',
  version="0.0.1",
  description='A simple tinygrad based project for inference with PFN\'s',
  url='https://github.com/jeroenprogrammeertaal/TinyPFN',
  author='Jeroen Taal',
  author_email='jeroen1749@gmail.com',
  license='MIT',
  packages=['tinypfn'],
  install_requires=["numpy", "tinygrad", "torch", "scikit-learn"],
  python_requires='>=3.8',
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License"
  ],
  include_package_data=True
)