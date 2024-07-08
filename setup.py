from setuptools import setup
from setuptools import find_packages

# change this.
NAME = "microformer-mgm"
AUTHOR = "Haohong Zhang"
EMAIL = "haohongzh@gmail.com"
URL = "https://github.com/HUST-NingKang-Lab/MGM"
LICENSE = "MIT"
DESCRIPTION = "MGM (Microbial General Model) as a large-scaled pretrained language model for interpretable microbiome data analysis."


if __name__ == "__main__":
    setup( 
        name=NAME,
        version="0.5.6",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        license=LICENSE,
        description=DESCRIPTION,
        packages=find_packages(),
        package_dir={'mgm': 'mgm'},
        include_package_data=True,
        install_requires=open("./requirements.txt", "r").read().splitlines(),
        long_description=open("./README.md", "r").read(),
        long_description_content_type='text/markdown',
        # change package_name to your package name.
        entry_points={
            "console_scripts": [
                "mgm=mgm.CLI:main"
            ]
        },
        package_data={
            # change package_name to your package name.
            # "mgm": ["resources/*", "resources/general_model/*"]
            "config": ["./resources/config.ini"],
            "general_model": ["./resources/general_model"],
			"phylo":["./resources/phylogeny.csv"],
            "MicroTokenizer":["./resources/MicroTokenizer.pkl"],
			"tmp":["./resources/tmp"]
        },
        zip_safe=True,
        classifiers=[
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Programming Language :: Python :: 3.9",
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Natural Language :: English"

        ],
        python_requires='>=3.9',
    )
