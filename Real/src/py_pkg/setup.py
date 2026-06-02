from setuptools import find_packages, setup

package_name = 'py_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nmp',
    maintainer_email='nmp@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mapodom_tf = py_pkg.mapodom_tf:main',
            'pursuit = py_pkg.pursuit:main',
            'path = py_pkg.path:main',
            'mux_vel = py_pkg.mux_vel:main',
            'slam = py_pkg.slam:main'
        ],
    },
)
