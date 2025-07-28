from glob import glob
from setuptools import find_packages, setup

package_name = 'vo_visualizer'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ament resource index
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        # package.xml and config.yaml into share/<package_name>
        ('share/' + package_name, ['package.xml', 'config.yaml']),
        # all .pth weights under share/<package_name>/weights/vo
        ('share/' + package_name + '/weights/vo',
         glob('weights/vo/*.pth')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'numpy',
        'torch',
    ],
    zip_safe=True,
    maintainer='park-ubuntu',
    maintainer_email='chansoo0710@gmail.com',
    description='Visualize VO trajectories and pointclouds in ROS2/RViz2',
    license='MIT',  # 예: MIT
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visualizer_node = vo_visualizer.visualizer_node:main',
            'dummy_node      = vo_visualizer.dummy_node:main',
        ],
    },
)
