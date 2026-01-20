"""
Utilidades para procesamiento de datos INSPIRE y catastro.

Este módulo contiene funciones para limpiar, clasificar y procesar
datos catastrales según los estándares INSPIRE de la UE.
"""

import re
import pandas as pd


def limpiar_caracteres(value):
    """
    Elimina caracteres de control no válidos de cadenas de texto.
    
    Args:
        value: Valor a limpiar (puede ser cualquier tipo).
    
    Returns:
        str o tipo original: Si es string, retorna el texto limpio.
                            Si no es string, retorna el valor original.
    
    Example:
        >>> limpiar_caracteres("texto\x00con\x1Fcaracteres")
        'textoconcaracteres'
    """
    if isinstance(value, str):
        # Reemplaza los caracteres no válidos con una cadena vacía
        return re.sub(r'[\x00-\x1F\x7F]', '', value)
    return value


def get_tipo(ucmc):
    """
    Clasifica el tipo de uso de una construcción según el código UCMC del catastro.
    
    Utiliza los códigos UCMC (Uso, Clase, Modalidad, Categoría) del catastro español
    para determinar el tipo de edificación.
    
    Args:
        ucmc: Código UCMC del catastro (puede ser int o str).
    
    Returns:
        str: Tipo de uso clasificado. Puede ser:
            - 'res_col_ab': Residencial colectivo abierto
            - 'res_col_cer': Residencial colectivo cerrado
            - 'res_col_gar': Residencial colectivo con garaje
            - 'res_uni_ais': Residencial unifamiliar aislado
            - 'res_uni_cer': Residencial unifamiliar cerrado
            - 'res_uni_gar': Residencial unifamiliar con garaje
            - 'industrial': Uso industrial
            - 'oficinas': Oficinas
            - 'comercial': Comercial
            - 'deportes': Deportivo
            - 'espectaculos': Espectáculos
            - 'ocio': Ocio y hostelería
            - 'salud': Sanitario
            - 'culturales': Cultural
            - 'singulares': Edificios singulares
            - 'otros': Otros usos
    
    Example:
        >>> get_tipo(111)
        'res_col_ab'
        >>> get_tipo('4001')
        'comercial'
    """
    ucmc_str = str(ucmc)
    if ucmc_str.startswith('111'): return 'res_col_ab'
    elif ucmc_str.startswith('112'): return 'res_col_cer'
    elif ucmc_str.startswith('113'): return 'res_col_gar'
    elif ucmc_str.startswith('121'): return 'res_uni_ais'
    elif ucmc_str.startswith('122'): return 'res_uni_cer'
    elif ucmc_str.startswith('123'): return 'res_uni_gar'
    elif ucmc_str.startswith('2'): return 'industrial'
    elif ucmc_str.startswith('3'): return 'oficinas'
    elif ucmc_str.startswith('4'): return 'comercial'
    elif ucmc_str.startswith('5'): return 'deportes'
    elif ucmc_str.startswith('6'): return 'espectaculos'
    elif ucmc_str.startswith('7'): return 'ocio'
    elif ucmc_str.startswith('8'): return 'salud'
    elif ucmc_str.startswith('9'): return 'culturales'
    elif ucmc_str.startswith('10'): return 'singulares'
    return 'otros'


def obtener_año(row):
    """
    Calcula el año de construcción o última reforma significativa de un inmueble.
    
    Prioriza el año de reforma (ar) si es válido (> 1000) y toma el máximo
    entre año de reforma y año de construcción (aec). Si no hay año de reforma
    válido, devuelve el año de construcción.
    
    Args:
        row: Fila de DataFrame que debe contener las columnas:
            - 'ar': Año de reforma
            - 'aec': Año efectivo de construcción
    
    Returns:
        float o None: Año de construcción o reforma (el más reciente),
                     o None si no hay datos válidos.
    
    Example:
        >>> row = pd.Series({'ar': 2010, 'aec': 1980})
        >>> obtener_año(row)
        2010
        >>> row = pd.Series({'ar': None, 'aec': 1980})
        >>> obtener_año(row)
        1980
    """
    ar, aec = row['ar'], row['aec']
    if pd.notna(ar) and ar > 1000:
        if pd.notna(aec):
            return max(ar, aec)
        return ar
    else:
        return aec