from setuptools import setup, find_packages
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()
setup(
    name='simplellm',
    version='0.1.0',    
    description='Build LLMs easily',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NikolayBlagoev/simplellm',
    author='Nikolay Blagoev',
    author_email='nickyblagoev@gmail.com',
    license='MIT License',
    install_requires=['datasets',
                      'torch',
                      'sentencepiece',
                      'requests',
                      'transformers'                 
                      ],
    classifiers=["Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ]
)