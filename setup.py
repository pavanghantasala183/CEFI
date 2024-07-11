from setuptools import setup, find_packages

setup(
    name='CEFI',
    version='0.1.0',
    description='Cluster-Enhanced Feature Importance for Unsupervised Learning',
    author='Pavan Ghantasala',
    author_email='pavanghantasala1@gmail.com',
    url='https://github.com/pavanghantasala183/CEFI',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'shap',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
