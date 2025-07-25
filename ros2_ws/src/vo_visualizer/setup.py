from setuptools import find_packages, setup

package_name = 'vo_visualizer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools',
                      'rclpy',
                      'numpy',
                      'torch'],
    zip_safe=True,
    maintainer='park-ubuntu',
    maintainer_email='chansoo0710@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visualizer_node = vo_visualizer.visualizer_node:main',
            'dummy_node = vo_visualizer.dummy_node:main',
        ],
    },
)
