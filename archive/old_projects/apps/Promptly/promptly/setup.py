from setuptools import setup, find_packages

setup(
    name='promptly',
    version='0.1.0',
    description='Promptly manage your prompts - A prompt library tool with versioning, branching, eval, and chaining',
    author='Your Name',
    py_modules=['promptly'],
    install_requires=[
        'click>=8.0.0',
        'PyYAML>=6.0',
    ],
    entry_points={
        'console_scripts': [
            'promptly=promptly:cli',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
