from setuptools import setup, find_packages

setup(
    name='cllm',
    version='0.1.0',  # Consider using semantic versioning
    description='A brief description of your project',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/cllm',  # Project home page or repository URL
    license='LICENSE',  # For example, 'MIT'
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        # List your project's dependencies here, e.g.,
        # 'numpy>=1.18.1',
        # 'pandas>=1.0.3',
    ],
    python_requires='>=3.10',  # Minimum Python version required
    keywords='your, project, keywords, here',  # Helps users find your project
    project_urls={
        'Documentation': 'https://github.com/yourusername/cllm',
        'Source': 'https://github.com/yourusername/cllm',
        'Tracker': 'https://github.com/yourusername/cllm/issues',
    },
    # If your package is a single module, use:
    # py_modules=["my_module"],
    
    # Entry points create executable commands and scripts
    entry_points={
        'console_scripts': [
            'cllm=cllm.cli:main',  # Example of a command-line interface entry point
        ],
    },
    # Include additional files into the package
    include_package_data=True,
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst', '*.md'],
        # And include any *.msg files found in the 'hello' package, too:
        'hello': ['*.msg'],
    },
)
