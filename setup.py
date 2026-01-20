from setuptools import setup, find_packages

setup(
    name='cpsv',  
    version='0.1.0',
    description='Utilidades para limpieza y análisis de datos',
    author='José Rojas-Quiroz',

    # Usa find_packages() para detectar automáticamente el paquete cpsv
    packages=find_packages(), 

    # Dependencias del paquete
    # Añadidas las dependencias de geoutils.py (osmnx, fuzzywuzzy)
    install_requires=[
        'pandas',
        'geopandas', 
        'matplotlib',
        'requests',
        'unidecode',
        'osmnx',
        'fuzzywuzzy',
        'python-Levenshtein',  # Para mejor rendimiento de fuzzywuzzy
    ],

    include_package_data=True,
    
    python_requires='>=3.7',
)