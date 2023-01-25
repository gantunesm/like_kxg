from setuptools import find_packages, setup
setup(
    name="like_kxg",
    version="0.0",
    author= "Gabriela Marques",
    author_email = 'gmarques@fsu.edu', 
    description="clgg and clkg likelihood",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["cobaya>=3.0"],
    package_data={"like_kxg": ["like_kxg.py"]},
)
