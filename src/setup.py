import setuptools
# import os

# with open(os.path.join(os.getcwd(), "../README.md"), "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="learn-diffeomorphism",  # Replace with your own username
    version="1.0.0",
    author="Bernardo Fichera",
    author_email="bernardo.fichera@gmail.com",
    description="NVP network for diffeomorphism learning",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/nash169/learn-diffeomorphism.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",                # math library
        "matplotlib",           # plotting library
    ],
    extras_require={
        "pytorch": [
            "torch",            # deep learning
            "torchvision",      # additional utilities
            "tensorboard"       # visualizations
        ],
        "dev": [
            "pylint",           # code quality tool
        ]
    },
)
