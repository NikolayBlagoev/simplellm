from setuptools import setup, find_packages

setup(
    name='simplellm',
    version='0.0.2',    
    description='Build LLMs easily',
    url='https://github.com/NikolayBlagoev/simplellm',
    author='Nikolay Blagoev',
    author_email='nickyblagoev@gmail.com',
    license='MIT License',
    install_requires=['datasets',
                      'torch',
                      'sentencepiece'                     
                      ],
    classifiers=["Programming Language :: Python :: 3",
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent",
                ]
)