from setuptools import setup, find_packages


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name='bone_xray',
    version='0.1.0',
    description='Bone-X-Ray project',
    author="Vasyl Borsuk, Mariia Kokshaikyna, Oleksandra Klochko",
    packages=find_packages(include=['bone_xray', 'bone_xray.*']),
    install_requires=parse_requirements("requirements.txt"),
    include_package_data=True,
)
