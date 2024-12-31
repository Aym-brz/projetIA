from setuptools import setup
import os
from glob import glob

package_name = 'projetIA'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
        (os.path.join('share', package_name, 'nodes'), glob('nodes/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aymeric Buriez',
    maintainer_email='aymeric.buriez@hotmail.fr',
    description='Using Gazebo Sim simulation with ROS to balance a double pendulum',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'speed_publisher = projetIA.speed_publisher:main',
            'state_subscriber = projetIA.state_subscriber:main',
            'world_control = projetIA.world_control:main',

        ],
    },
)