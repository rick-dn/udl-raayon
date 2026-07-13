from setuptools import setup

package_name = 'detection_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='udl',
    maintainer_email='hello@usefuldynamics.io',
    description='Person detection node using MobileNet SSD TFLite',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_detector = detection_pkg.person_detector_node:main',
            'green_detector = detection_pkg.green_detector_node:main',
            'aruco_detector = detection_pkg.aruco_detector_node:main',
        ],
    },
)
