from setuptools import setup

setup(
    name='limpiataclia',  
    version='0.1.0',
    description='Librería de utilidades para limpieza de datos inmobiliarios de BCN',
    author='José Rojas-Quiroz',

    # Esta parte es la CLAVE para archivos sueltos:
    # Indica el nombre del archivo .py (sin la extensión) que quieres exponer
    py_modules=['limpiataclia'], 

    # Aquí definimos qué librerías necesita tu código para funcionar.
    # He puesto las más comunes, pero revisa si el archivo usa otras.
    install_requires=[
        'pandas',
        #'numpy',
        'geopandas', 
        'matplotlib',
        #'shapely',
        #'langdetect',
        #'deep_translator',
        #'pysentimiento',
        #'zipfile',
        'requests',
        #'io',
        #'pathlib',
        'unidecode' 
    ],

    include_package_data=True,
)