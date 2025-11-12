from setuptools import find_packages, setup

package_name = 'kumdori_rag_ver1'

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
    maintainer='khy',
    maintainer_email='yonghun@rastech.co.kr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'kumdori_chatbot_node = kumdori_rag_ver1.kumdori_chatbot:main',
        ],
    },
)
