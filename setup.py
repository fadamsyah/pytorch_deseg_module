from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'Detection and Segmentation Module'
LONG_DESCRIPTION = 'The object detection module is obtained from Zyolo17 EfficientDet where the segmentation module is obtained from Inside-Outside-Guidance'

# Setting up
setup(
    name='pytorch_deseg_module',
    version=VERSION,
    author='Fadillah Adamsyah Maani',
    author_email='fadillahadam11@gmail.com',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    
    keywords=['python', 'pytorch', 'deep learning', 'object detection',
              'semantic segmentation', 'Zyolo17 Efficientdet',
              'Inside-Outside-Guidance'],
    classifiers=[
        "Intended Audience :: Any",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
