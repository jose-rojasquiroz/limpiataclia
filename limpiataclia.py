#####OJO: Aún falta añadir los procesos que permiten obtener los gpkg que se utilizan aquí (Ctrl+F para ver cuáles son)

from pathlib import Path
import pandas as pd
#import numpy as np
import re
#import unicodedata
from unidecode import unidecode
import geopandas as gpd
from shapely.geometry import Point
from langdetect import detect
from deep_translator import GoogleTranslator
from pysentimiento import create_analyzer #para el analisis de sentimiento


### Para leer los datos
import zipfile
import io
import requests

URL_DEL_ZIP = "https://github.com/jose-rojasquiroz/limpiataclia/releases/download/0.1.0/data.zip"
DATA_DIR = Path.home() / '.limpiataclia_data' # Crear un directorio oculto en el home del usuario

def asegurar_datos():
    """Descarga y descomprime los datos si no existen."""
    # Si la carpeta ya existe y tiene contenido, asumimos que ya se bajó
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        return

    print(f"Descargando datos necesarios (aprox 170MB)... Esto solo ocurre la primera vez.")
    print(f"Guardando en: {DATA_DIR}")
    
    try:
        # Descargar el zip
        r = requests.get(URL_DEL_ZIP)
        r.raise_for_status() # Lanza error si el link está mal
        
        # Descomprimir en memoria
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(DATA_DIR)
        print("¡Datos descargados y listos!")
        
    except Exception as e:
        print(f"Error descargando los datos: {e}")
        # Opcional: Borrar carpeta corrupta si falla
        raise

# 3. Ejecutamos la comprobación ANTES de que el resto del script necesite los datos
asegurar_datos()

# --- FIN DEL BLOQUE NUEVO ---

# Directorio de datos
DATA_DIR = Path(__file__).resolve().parent / 'data'

def extract_coordinates(df): 
    """
    Extrae las coordenadas de manera segura, manejando casos problemáticos
    """
    # Eliminar filas con valores problemáticos
    df = df[~df['Lon/Lat'].str.contains('Your Plan is insufficient', na=False)]
    
    # Extraer coordenadas
    df['lat'] = pd.to_numeric(
        df['Lon/Lat'].str.extract(r'([\d.]+),')[0], 
        errors='coerce'
    )
    
    df['lon'] = pd.to_numeric(
        df['Lon/Lat'].str.extract(r'VGPSLon\\":([\d.]+)')[0], 
        errors='coerce'
    )
    
    # Eliminar filas con coordenadas inválidas
    df = df.dropna(subset=['lat', 'lon'])
    
    return df

def extract_property_code(df):
    """
    Extrae el código numérico del inmueble solo si la columna es string.
    """
    if df['codigo_inmueble'].dtype == 'object':
        df['codigo_inmueble'] = df['codigo_inmueble'].str.extract('(\d+)').astype(int)
    return df

def remove_duplicates_by_code(df):
    """
    Elimina duplicados según criterios específicos:
    1. Por código de inmueble
    """
    print(f"\nEliminando duplicados...")
    print(f"Registros iniciales: {len(df)}")
    
    # Duplicados por código de inmueble
    df = df.drop_duplicates(subset=['codigo_inmueble'], keep='first')
    print(f"Tras eliminar duplicados por código de inmueble: {len(df)}")
    
    return df

def remove_duplicates_by_descrip(df, tipo_oferta='vivienda'):
    """
    Elimina duplicados según criterios específicos:
    1. Por descripción y superficie
    2. Por descripción y número de habitaciones (solo para viviendas)
    """
    print(f"\nEliminando duplicados...")
    print(f"Registros iniciales: {len(df)}")
    
    # Duplicados por descripción y superficie
    df = df.drop_duplicates(subset=['Description', 'superficie'], keep='first')
    print(f"Tras eliminar duplicados por descripción y superficie: {len(df)}")
    
    # Duplicados por descripción y habitaciones (solo para viviendas)
    if tipo_oferta == 'vivienda':
        df = df.drop_duplicates(subset=['Description', 'n_habs'], keep='first')
        print(f"Tras eliminar duplicados por descripción y habitaciones: {len(df)}")
    
    return df

def remove_spatial_duplicates(processed, grid_path, tipo_oferta=None):
    """
    Elimina duplicados espaciales dentro de una misma celda de cuadrícula ('id')
    según combinaciones de campos que dependen del tipo de oferta.
    """
    print("\nAplicando filtro de duplicados espaciales por cuadrícula...")

    grid = gpd.read_file(grid_path)
    grid = grid.to_crs(processed.crs)

    # 🧹 Eliminar columnas problemáticas antes del join
    for col in ['index_right', 'index_left']:
        if col in processed.columns:
            processed = processed.drop(columns=col)

    joined = gpd.sjoin(processed, grid[['id', 'geometry']], how='left', predicate='within')

    if 'id' not in joined.columns:
        raise ValueError("No se encontró la columna 'id' tras el join con la cuadrícula.")

    print(f"Join espacial completado: {joined['id'].notna().sum()} puntos asociados a cuadrícula.")

    def marcar_duplicados(df):
        duplicados = pd.Series(False, index=df.index)

        # descripcion + superficie → para todos los tipos
        if {'descripcion', 'superficie'}.issubset(df.columns):
            duplicados |= df.duplicated(subset=['descripcion', 'superficie'], keep='first')

        # descripcion + n_habs → solo para viviendas
        if tipo_oferta == 'vivienda' and {'descripcion', 'n_habs'}.issubset(df.columns):
            duplicados |= df.duplicated(subset=['descripcion', 'n_habs'], keep='first')

        # superficie + precio → para todos los tipos
        if {'superficie', 'precio_euros'}.issubset(df.columns):
            duplicados |= df.duplicated(subset=['superficie', 'precio_euros'], keep='first')

        return df.loc[~duplicados]


    cleaned = (
    joined.groupby('id', group_keys=False)
    .apply(marcar_duplicados, include_groups=False)
    .drop(columns=['index_right', 'id'], errors='ignore')
)

    return cleaned

# Funciones de limpieza de texto
def clean_text(text):
    """
    Limpia y normaliza el texto aplicando múltiples transformaciones.
    """
    if not isinstance(text, str):
        return ''
        
    text = text.replace(u'\xa0', u' ')
    
    # Normalización de unidades y símbolos
    text = re.sub(r'(m2)|(m²)', ' m²', text)
    text = re.sub(r'(euro[^s])|(euros)|(€)', ' euros', text)
    
    # Eliminación de información sensible/irrelevante
    patterns = {
        'iban': r'ES\d{2}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{2}|ES\d{20}|ES[ ]\d{2}[ ]\d{3}[ ]\d{3}[ ]\d{3}[ ]\d{5}',
        'email': r'(?:(?!.*?[.]{2})[a-zA-Z0-9](?:[a-zA-Z0-9.+!%-]{1,64}|)|\"[a-zA-Z0-9.+!% -]{1,64}\")@[a-zA-Z0-9][a-zA-Z0-9.-]+(.[a-z]{2,}|.[0-9]{1,})',
        'phone': r'(?:(?:\+|00)33[\s.-]{0,3}(?:[\s.-]{0,3})?|0)[1-9](?:(?:[\s.-]?\d{2}){4}|\d{2}(?:[\s.-]?\d{3}){2})|(\d{2}[ ]\d{2}[ ]\d{3}[ ]\d{3})',
        'website': r'(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z][a-z\-]*',
        'large_numbers': r'\b\d{4,}\b',
        'refs': r'\(.*?\)',
        'brackets': r' [^]*\]|\{[^}]*\}',
    }
    
    for pattern in patterns.values():
        text = re.sub(pattern, '', text)
    
    # Limpieza general
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Espacios entre números
    text = re.sub(r'\*|\-{2,}|\s{2,}|[¡!]+', ' ', text)  # Caracteres especiales
    text = text.strip()
    
    return text

# Funciones de extracción de características
def extract_floor_info(text, characteristics):
    """
    Extrae información sobre el piso/planta del inmueble.
    Considera tanto la descripción general como las características.
    """
    def get_floor_number(text):
        if pd.isna(text):
            return 999
            
        text = str(text).lower()  # Aseguramos que text sea string
        text = unidecode(text)
        
        # Diccionario de palabras numéricas
        num_dict = {
            'primera': 1, 'primero': 1, 'primer': 1, '1ra': 1,
            'segunda': 2, 'segundo': 2, '2da': 2,
            'tercera': 3, 'tercero': 3, 'tercer': 3,
            'cuarta': 4, 'cuarto': 4,
            'quinta': 5, 'quinto': 5,
            'sexta': 6, 'sexto': 6,
            'septima': 7, 'septimo': 7,
            'octava': 8, 'octavo': 8,
            'novena': 9, 'noveno': 9,
            'decima': 10, 'decimo': 10
        }
        
        # Casos especiales
        if any(term in text for term in ['planta baja', 'bajo']):
            return 0
        if 'sotano' in text:
            return -2
        if any(term in text for term in ['semisotano', 'semi-sotano']):
            return -1
            
        # Búsqueda por patrones
        for word, num in num_dict.items():
            if any(pattern in text for pattern in [
                f'planta {word}', f'{word} planta',
                f'piso {word}', f'{word} piso'
            ]):
                return num
                
        # Búsqueda numérica
        for i in range(21):
            if any(pattern in text for pattern in [
                f'planta {i}', f'{i}ª planta',
                f'{i}º planta', f'{i} planta'
            ]):
                return i
                
        return 999
    
    # Asegurar que text y characteristics son strings
    text = str(text) if not pd.isna(text) else ''
    characteristics = str(characteristics) if not pd.isna(characteristics) else ''
    
    # Intenta primero con características generales
    floor = get_floor_number(characteristics)
    
    # Si no encuentra, busca en la descripción
    if floor == 999:
        floor = get_floor_number(text)
        
    # Verifica si es ático
    if floor == 999:
        combined_text = text.lower() + ' ' + characteristics.lower()
        if any(term in combined_text for term in ['atico', 'aticos']):
            floor = 55
            
    return floor


def extract_property_features(row):
    """
    Extrae características de la propiedad: terraza, terraza_m2, terraza_grande, parking, trastero, ascensor, ac, calefacción y piscina.
    """
    result = pd.Series({
        'terraza': 0,
        'terraza_m2': 0,
        'parking': 0,
        'trastero': 0,
        'ascensor': 0,
        'ac': 0,
        'calefaccion': 0,
        'piscina': 0
    })
        
    # TERRAZA: Primero revisar Distribution
    if not pd.isna(row['Distribution']):
        distribution = row['Distribution'].lower()
        terrace_match = re.search(r'(terraza|balc[oó]n).*?(\d+)\s?m2', distribution)
        if terrace_match:
            result['terraza'] = 1
            result['terraza_m2'] = int(terrace_match.group(2))
        elif re.search(r'(terraza|balc[oó]n)', distribution):
            result['terraza'] = 1
            result['terraza_m2'] = 1

    # Si no se encontró terraza en Distribution, buscar en Description
    if result['terraza'] == 0 and not pd.isna(row['Description']):
        description = row['Description'].lower()
        if re.search(r'(terraza|balc[oó]n)', description):
            result['terraza'] = 1
            terrace_m2_match = re.search(r'(\d+)\s?m2', description)
            result['terraza_m2'] = int(terrace_m2_match.group(1)) if terrace_m2_match else 1

    # VALIDACIÓN DEL TAMAÑO DE TERRAZA (aplicar siempre que terraza == 1)
    if result['terraza'] == 1 and result['terraza_m2'] > 0 and row['superficie'] > 0:
        terraza_umbral = 0.3
        terraza_nuevo_pct = 0.05
        
        proporcion_terraza = result['terraza_m2'] / row['superficie']
        
        if proporcion_terraza >= terraza_umbral:
            nuevo_valor = row['superficie'] * terraza_nuevo_pct
            result['terraza_m2'] = int(nuevo_valor)  # o float(nuevo_valor)

    # TERRAZA_GRANDE
    result['terraza_grande'] = 1 if result['terraza_m2'] >= 20 else 0

    # PARKING
    if not pd.isna(row['Description']):
        desc = row['Description'].lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        
        included_patterns = [
            r"(plaza|espacio|estacionamiento|garaje).*incluida.*precio",
            r"disponible.*plaza.*parking.*incluida.*precio",
            r"plaza.*parking.*incluida.*precio",
            r"parking.incluida.*en.*el.*precio",
            r"aparcamiento.incluido.*en.*el.*precio",
            r"incluye.*plazas?.*aparcamientos?",
            r"la.*propiedad.*incluye.*(plaza|plazas?).*aparcamiento",
            r"el.*piso.*incluye.*(plaza|plazas?).*aparcamiento",
            r"el.*piso.*cuenta.*(plaza|plazas?).*parking",
            r"parking.*\(incluido en el precio\)",
            r"aparcamiento.*\(incluido en el precio\)"
        ]
        
        optional_patterns = [
            r"(adquirir|posible).*opcional.*parking",
            r"acceso.*parking.*incluido.*por.*\d+.*€",
            r"plaza.*parking.*adicional",
            r"plaza.*aparcamiento.*adicional",
            r"posibilidad de.*plaza.*parking",
            r"posibilidad de.*plaza.*aparcamiento"
        ]
        
        if any(re.search(pattern, desc) for pattern in included_patterns):
            result['parking'] = 2
        elif any(re.search(pattern, desc) for pattern in optional_patterns):
            result['parking'] = 1
    
    # TRASTERO
    # Primero revisar General Characteristics
    if not pd.isna(row['General Characteristics']):
        gen_char = row['General Characteristics'].lower()
        if 'trastero' in gen_char or 'trasteros' in gen_char:
            result['trastero'] = 2
    
    # Si no se encontró trastero, revisar Description
    if result['trastero'] == 0 and not pd.isna(row['Description']):
        desc = row['Description'].lower()
        
        trastero_privado_patterns = [
            r"trastero.*privado",
            r"propiedad incluye.*trastero.*finca",
            r"trastero.*incluso.*misma finca",
            r"cuenta con.*trastero",
            r"trastero.*\(incluido en el precio\)"
        ]
        
        trastero_comunitario_patterns = [
            r"trastero.*comunitario",
            r"finca.*trastero.*comunitario",
            r"acceso.*trastero.*comunitario",
            r"trastero.*comunidad"
        ]
        
        if any(re.search(pattern, desc) for pattern in trastero_privado_patterns):
            result['trastero'] = 2
        elif any(re.search(pattern, desc) for pattern in trastero_comunitario_patterns):
            result['trastero'] = 1
    
    # ASCENSOR, AC, CALEFACCIÓN Y PISCINA
    # Primero revisar Equipment
    if not pd.isna(row['Equipment']):
        equip = row['Equipment'].lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        
        # Ascensor
        if not any(pattern in equip for pattern in ['sin ascensor', 'no tiene ascensor']):
            result['ascensor'] = 1 if 'ascensor' in equip else 0
            
        # AC
        if not any(pattern in equip for pattern in ['sin aire acondicionado', 'sin a/c']):
            result['ac'] = 1 if any(pattern in equip for pattern in ['aire acondicionado', 'a/c', 'climatizador']) else 0
            
        # Calefacción
        if not any(pattern in equip for pattern in ['sin calefaccion', 'no tiene calefaccion']):
            result['calefaccion'] = 1 if any(pattern in equip for pattern in ['calefaccion', 'caldera', 'radiadores']) else 0
            
        # Piscina
        result['piscina'] = 1 if 'piscina' in equip else 0
    
    # Si no se encontró información en Equipment, revisar Description
    desc = row['Description'].lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u') if not pd.isna(row['Description']) else ''
    
    if result['ascensor'] == 0:
        result['ascensor'] = 1 if ('ascensor' in desc and not any(pattern in desc for pattern in ['sin ascensor', 'no tiene ascensor'])) else 0
        
    if result['ac'] == 0:
        ac_patterns = ['aire acondicionado', r'\ba/c\b', 'ac por conductos', 'climatizador', 'sistema de aire', 'bomba de calor']
        result['ac'] = 1 if (any(re.search(pattern, desc) for pattern in ac_patterns) and 
                            not any(pattern in desc for pattern in ['sin aire acondicionado', 'sin a/c'])) else 0
        
    if result['calefaccion'] == 0:
        heating_patterns = ['calefaccion', 'calefactores?', 'caldera', 'calefaccion central', 'radiadores?', 'calefaccion por suelo radiante']
        result['calefaccion'] = 1 if (any(re.search(pattern, desc) for pattern in heating_patterns) and
                                    not any(pattern in desc for pattern in ['sin calefaccion', 'no tiene calefaccion'])) else 0
    
    if result['piscina'] == 0:
        result['piscina'] = 1 if 'piscina' in desc else 0
    
    return result



########Funciones que sirven para process_basic_info
# Función para extraer la información del precio y otras características
def extraer_info(texto):
    rooms_match = re.search(r'(\d+)\s*(habitaciones|habitaci[oó]n)', texto)
    n_rooms = int(rooms_match.group(1)) if rooms_match else 0
    return pd.Series([n_rooms])

def extraer_info_description(texto):
    rooms_match = re.search(r'(\d+)\s*(?:habitacion(?:es)?)\b', str(texto))
    n_rooms = int(rooms_match.group(1)) if rooms_match else 0
    return n_rooms

def extraer_info_description2(texto):
    # Buscar números, palabras numéricas o "una" seguidos de "habitación", "habitaciones" o "habitación" en la descripción
    rooms_match = re.search(r'(?:\b(una|uno|dos|tres|cuatro|cinco|seis|siete|ocho|nueve|diez|once|doce|trece|catorce|quince|dieciséis|diecisiete|dieciocho|diecinueve|veinte)\b|\d+)\s*habitacion(?:es)?\b', str(texto))
    # Convertir palabras numéricas a números
    if rooms_match and rooms_match.group(1):
        num_word = rooms_match.group(1)
        num_dict = {'una': 1, 'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5, 'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
                    'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14, 'quince': 15, 'dieciséis': 16, 'diecisiete': 17, 'dieciocho': 18,
                    'diecinueve': 19, 'veinte': 20}
        n_rooms = num_dict[num_word]
    else:
        # Extraer solo el número encontrado
        n_rooms = int(re.search(r'\d+', rooms_match.group(0)).group()) if rooms_match else 0
    return n_rooms

def extraer_info_baño(texto):
    bath_match = re.search(r'\b(\d+|[a-zA-Z]+)\s*(baño|baños)\b', str(texto), re.IGNORECASE)
    if bath_match:
        num_baths = bath_match.group(1)
        if num_baths.isdigit():
            n_baths = int(num_baths)
        else:
            num_dict = {'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5}
            n_baths = num_dict.get(num_baths.lower(), 1)
    else:
        n_baths = 0
    return n_baths

def extraer_info_baño2(texto):
    # Buscar la palabra "baño" en la descripción
    bath_match = re.search(r'\b(baño|baño completo)\b', str(texto), re.IGNORECASE)

    # Si se encuentra la palabra "baño"
    if bath_match:
        return 1  # Si encuentra la palabra "baño", establecemos n_baths en 1
    else:
        return 0  # Si no se encuentra la palabra "baño", establecemos n_baths en 0
###############

def process_basic_info(row):
    """
    Procesa información básica del inmueble con validación adicional de habitaciones.
    """
    info = {
        'superficie': 0,
        'n_habs': 0,
        'n_baños': 0,
        'precio_euros': 0
    }
    
    # Superficie y habitaciones desde Price_other
    price_other = str(row['Price_other']) if pd.notna(row['Price_other']) else ''
    superficie_match = re.search(r'(\d+)\s*m2', price_other)
    habs_match = re.search(r'(\d+)\s*habitaci', price_other)
    
    info['superficie'] = int(superficie_match.group(1)) if superficie_match else 0
    info['n_habs'] = int(habs_match.group(1)) if habs_match else 0
    
    # Validación adicional de habitaciones
    if info['n_habs'] == 0 or info['n_habs'] > 10:
        info['n_habs'] = extraer_info(str(row['Price_other']) if pd.notna(row['Price_other']) else '')[0]
        if info['n_habs'] == 0 or info['n_habs'] > 10:
            info['n_habs'] = extraer_info(str(row['Distribution']) if pd.notna(row['Distribution']) else '')[0]
            if info['n_habs'] == 0 or info['n_habs'] > 10:
                info['n_habs'] = extraer_info_description(str(row['Description']) if pd.notna(row['Description']) else '')
                if info['n_habs'] == 0 or info['n_habs'] > 10:
                    info['n_habs'] = extraer_info_description2(str(row['Description']) if pd.notna(row['Description']) else '')

    info['n_habs'] = int(info['n_habs'])

    # Si el número tiene más de un dígito, quedarse con el primero
    if info['n_habs'] >= 10:
        info['n_habs'] = int(str(info['n_habs'])[0])
    
    info['n_habs'] = int(info['n_habs'])


    # Baños: primero buscar en Price_other
    po_baños = re.search(r'(\d+)\s*baño', str(row['Price_other']))
    info['n_baños'] = int(po_baños.group(1)) if po_baños else 0
    
    # Si es 0, buscar en Distribution
    if info['n_baños'] == 0:
        dist_baños = re.search(r'(\d+)\s*baño', str(row['Distribution']))
        info['n_baños'] = int(dist_baños.group(1)) if dist_baños else 0
    
    # Si es 0, aplicar extraer_info_baño en Description
    if info['n_baños'] == 0:
        info['n_baños'] = extraer_info_baño(row['Description'])
        
    # Si sigue siendo 0, aplicar extraer_info_baño2
    if info['n_baños'] == 0:
        info['n_baños'] = extraer_info_baño2(row['Description'])
    
    # Si el valor tiene dos dígitos, quedarse con el primero
    if info['n_baños'] >= 10:
        info['n_baños'] = int(str(info['n_baños'])[0])
    
    # Si aún es 0, convertir a 1 por defecto
    if info['n_baños'] == 0:
        info['n_baños'] = 1
    
    # Precio: procesar solo si no es numérico
    try:
        info['precio_euros'] = pd.to_numeric(row['precio_euros'])
    except:
        precio = str(row['precio_euros']).replace('.', '').replace('€', '').replace(' ', '')
        info['precio_euros'] = int(precio) if precio.isdigit() else 0
    
    return pd.Series(info)

def process_epc_rating(epc_text):
    """
    Procesa la calificación energética, retornando la letra original declarada,
    la letra calculada según valores numéricos, y si coinciden.
    Mapea las letras a valores numéricos (A=7, ..., G=1).
    Para valores vacíos:
    - EPC_e_original = 1 (equivalente a G)
    - EPC_e_calculado = 999 (valor especial para indicar ausencia de dato)
    """
    # Mapping de letras a valores numéricos
    mapping = {
        'A': 7,
        'B': 6,
        'C': 5,
        'D': 4,
        'E': 3,
        'F': 2,
        'G': 1
    }
    
    # Inicializar los resultados para el caso de valor vacío
    if pd.isna(epc_text) or str(epc_text).strip() == '':
        return pd.Series({
            'EPC_e_original': 1,  # Equivalente a G
            'EPC_e_calculado': 999,  # Valor especial para ausencia de dato
            'EPC_coincidente': 0
        })
    
    # Inicializar los resultados como Series con valores por defecto
    result = pd.Series({
        'EPC_e_original': 'G',
        'EPC_e_calculado': 'G',
        'EPC_coincidente': 0
    })
    
    # Obtener letra original declarada
    letter_match = re.search(r'Emisiones:\s*([A-G])', str(epc_text), re.IGNORECASE)
    if letter_match:
        result['EPC_e_original'] = letter_match.group(1).upper()
    
    # Obtener valor numérico y calcular letra correspondiente
    value_match = re.search(r'(\d+(\.\d+)?)\s*kg CO2 m2\s*/\s*año', str(epc_text))
    if value_match:
        value = float(value_match.group(1))
        # Asignar letra según el rango de valores
        if value <= 6.1:
            result['EPC_e_calculado'] = 'A'
        elif value <= 9.9:
            result['EPC_e_calculado'] = 'B'
        elif value <= 15.3:
            result['EPC_e_calculado'] = 'C'
        elif value <= 23.5:
            result['EPC_e_calculado'] = 'D'
        elif value <= 49:
            result['EPC_e_calculado'] = 'E'
        elif value <= 57.3:
            result['EPC_e_calculado'] = 'F'
        else:
            result['EPC_e_calculado'] = 'G'
    
    # Convertir y mapear las letras a valores numéricos
    result['EPC_e_original'] = mapping[result['EPC_e_original']]
    result['EPC_e_calculado'] = mapping[result['EPC_e_calculado']]
    
    # Verificar coincidencia
    result['EPC_coincidente'] = 1 if result['EPC_e_original'] == result['EPC_e_calculado'] else 0
    
    return result

def detect_housing_type_venta(row):
    """
    Detecta el tipo de vivienda y si es plurifamiliar.
    Retorna un diccionario con el tipo y la clasificación plurifamiliar.
    """
    # Obtener primera palabra del título
    def get_first_word(title):
        if pd.isna(title):
            return None
        words = title.split()
        if words:
            return unidecode(words[0].lower())
        return None
    
    result = {
        'tipo': None,
        'plurifam': 1  # Por defecto es plurifamiliar
    }
    
    # Primero ver si es un ático
    if row['planta'] == 55 or any(term in row['Title'].lower() for term in ['atico', 'aticos']):
        result['tipo'] = 'atico'
    else:
        # Si no es ático, detectar tipo por primera palabra del título
        tipo = get_first_word(row['Title'])
        
        # Convertir tipos similares
        if tipo in ['chalet', 'masia', 'torre']:
            tipo = 'casa'
        elif tipo in ['apartamento', 'planta']:
            tipo = 'piso'
        
        result['tipo'] = tipo
    
    # Asignar plurifam basado solo en si es casa
    if result['tipo'] == 'casa':
        result['plurifam'] = 0
    
    return result

def detect_housing_type_alquiler(row):
    """
    Detecta el tipo de vivienda y si es plurifamiliar.
    Retorna un diccionario con el tipo y la clasificación plurifamiliar.
    """
    # Obtener primera palabra del título
    def get_second_word(title):
        if pd.isna(title):
            return None
        words = title.split()
        if len(words) < 2:
            return None
        return unidecode(words[1].lower())
    
    result = {
        'tipo': None,
        'plurifam': 1  # Por defecto es plurifamiliar
    }
    
    # Primero ver si es un ático
    if row['planta'] == 55 or any(term in row['Title'].lower() for term in ['atico', 'aticos']):
        result['tipo'] = 'atico'
    else:
        # Si no es ático, detectar tipo por primera palabra del título
        tipo = get_second_word(row['Title'])
        
        # Convertir tipos similares
        if tipo in ['chalet', 'masia', 'torre']:
            tipo = 'casa'
        elif tipo in ['apartamento', 'planta']:
            tipo = 'piso'
        
        result['tipo'] = tipo
    
    # Asignar plurifam basado solo en si es casa
    if result['tipo'] == 'casa':
        result['plurifam'] = 0
    
    return result

def detect_occupancy_status(text):
    """
    Detecta si la vivienda está ocupada o alquilada.
    """
    if pd.isna(text):
        return {'okupa': 0, 'alquilado': 0}
        
    text = text.lower()
    
    # Keywords para okupación
    okupa_keywords = [
        'okupa', 'okupado', 'ocupado ilegalmente', 'ocupación ilegal',
        'desalojo', 'squatter', 'ocupas', 'okupas'
    ]
    
    # Keywords para alquiler
    alquiler_keywords = [
        'alquilado', 'arrendado', 'en alquiler', 'inquilino actual',
        'rentado', 'con inquilinos', 'renta actual'
    ]
    
    return {
        'okupa': 1 if any(keyword in text for keyword in okupa_keywords) else 0,
        'alquilado': 1 if any(keyword in text for keyword in alquiler_keywords) else 0
    }

def check_habitability_certificate(text):
    """
    Verifica si la vivienda tiene cédula de habitabilidad.
    """
    if pd.isna(text):
        return 1  # Por defecto asumimos que tiene cédula
        
    text = unidecode(text.lower())
    
    # Patrones negativos
    negative_patterns = [
        r"no tiene.*cedula",
        r"sin.*cedula",
        r"no dispone.*cedula",
        r"cedula.en.*tramite",
        r"cedula.por.*obtener"
    ]
    
    # Patrones positivos
    positive_patterns = [
        r"cedula de habitabilidad",
        r"dispone.*cedula",
        r"tiene.*cedula"
    ]
    
    if any(re.search(pattern, text) for pattern in negative_patterns):
        return 0
    if any(re.search(pattern, text) for pattern in positive_patterns):
        return 1
    return 1  # Si no encontramos nada, asumimos que tiene cédula

def process_text_language(text):
    """
    Detecta el idioma del texto y lo traduce si no es español.
    Se divide en dos partes el procesamiento para no saturar la API.
    """
    if pd.isna(text) or text.strip() == '':
        return {'text': '', 'lang': 'es'}
    
    try:
        # Detectar idioma
        lang = detect(text)
        
        # Si ya está en español, devolver el texto original
        if lang == 'es':
            return {'text': text, 'lang': lang}
        
        # Si no está en español, traducir (dividiendo el texto en dos partes)
        text_length = len(text)
        halfway_point = text_length // 2
        
        # Partir el texto en dos mitades
        first_half = text[:halfway_point]
        second_half = text[halfway_point:]
        
        # Traducir cada mitad
        translator = GoogleTranslator(source='auto', target='es')
        translated_first = translator.translate(first_half)
        translated_second = translator.translate(second_half)
        
        # Combinar las traducciones
        full_translation = ' '.join([translated_first, translated_second])
        
        return {'text': full_translation, 'lang': lang}
        
    except:
        return {'text': text, 'lang': 'es'}  # En caso de error, devolver el texto original

def analyze_sentiment(text):
    """
    Analiza el sentimiento del texto usando pysentimiento.
    """
    if pd.isna(text) or text.strip() == '':
        return {'POS': 0, 'NEU': 0, 'NEG': 0}
        
    try:
        analyzer = create_analyzer(task="sentiment", lang="es")
        result = analyzer.predict(text)
        return {
            'POS': result.probas['POS'],
            'NEU': result.probas['NEU'],
            'NEG': result.probas['NEG']
        }
    except:
        return {'POS': 0, 'NEU': 0, 'NEG': 0}

def has_obra_nueva(row):
    """
    Detecta si es obra nueva y aplica filtros específicos.
    Retorna:
        int: 1 si es obra nueva, 0 si no lo es
    """
    text = ' '.join([
        str(row['Description']),
        str(row['Title']),
        str(row['texto_destacado'])
    ]).lower()
    
    return int(bool(re.search(r'obra nueva|planta nueva|nueva planta|nova planta|obra nova', text)))


def clean_gdf_for_join(gdf):
    """
    Limpia un GeoDataFrame para prepararlo para joins espaciales.
    """
    # Lista de columnas a eliminar si existen
    cols_to_drop = ['index_right', 'index_left', 'index']
    
    cleaned = gdf.copy()
    # Eliminar columnas si existen
    for col in cols_to_drop:
        if col in cleaned.columns:
            cleaned = cleaned.drop(columns=[col])
    
    # Reset del índice
    cleaned = cleaned.reset_index(drop=True)
    return cleaned

#########  PROMEDIOS ##########
def update_floor_info(gdf, alturas_gdf, tipo_oferta='vivienda'):
    """
    Incorpora información de plantas usando datos catastrales (ya no usamos la declarada en las ofertas, por ser muy difícil de captar).
    Limita las alturas consideradas a un máximo de 20 plantas.
    Para casos sin buffer, usa la planta dividida sobre 2, redeondeado a 0 decimales.
    Para casos con buffer, usa el promedio de plantas dividido sobre 2, redondeado a 0 decimales.
    
    Para locales comerciales y oficinas, solo se consideran casos especiales donde planta=999 o planta=55.
    """
    print("Preparando datos...")
    # Crear copias limpias y filtrar alturas
    processed = gdf.copy()
    alturas = alturas_gdf[['geometry', 'pt']].copy()
    alturas = alturas[alturas['pt'] <= 20].copy()  # Filtrar edificios de más de 20 plantas
    
    # Reset índices
    processed = processed.reset_index(drop=True)
    alturas = alturas.reset_index(drop=True)
    
    if tipo_oferta == 'vivienda':
        print(f"Edificios considerados (pt <= 20): {len(alturas)}")
    else:
        print(f"Parcelas consideradas (local de oficina o comercio con planta <=19): {len(alturas)}")
  
    # Identificar casos especiales según tipo de oferta
    if tipo_oferta == 'vivienda':
        mask_especial = (
            (processed['planta'] == 999) |
            (processed['planta'] == 55) |
            (processed['tipo']=='atico')|
            (processed['tipo'] == 'duplex') |
            (processed['tipo'] == 'triplex') |
            (processed['planta']<0)|
            ((processed['obra_nueva'] == 1) & (processed['planta'] < 1) & (processed['plurifam'] == 1))
        )
    else:
        mask_especial = (
            (processed['planta'] == 999) |
            (processed['planta'] == 55)
        )

    casos_especiales = processed[mask_especial].copy()
    print(f"Casos especiales identificados: {len(casos_especiales)}")
    
    if len(casos_especiales) == 0:
        return processed
    
    try:
        # 1. Join directo
        print("Realizando join directo...")
        join_result = gpd.sjoin(
            casos_especiales[['geometry', 'planta']],
            alturas,
            how='left',
            predicate='within'
        )
        
        # Actualizar plantas para casos con match directo
        matched = join_result[pd.notnull(join_result['pt'])]
        print(f"Casos con match directo: {len(matched)}")
        
        for idx in matched.index:
            processed.loc[idx, 'planta'] = round(matched.loc[idx, 'pt'] / 2, 0)
        
        # 2. Buffer para casos sin match
        unmatched = casos_especiales.index.difference(matched.index)
        if len(unmatched) > 0:
            print(f"Intentando match con buffer para {len(unmatched)} casos...")
            buffer_cases = casos_especiales.loc[unmatched].copy()
            buffer_cases.geometry = buffer_cases.geometry.buffer(300)
            
            buffer_join = gpd.sjoin(
                buffer_cases[['geometry', 'planta']],
                alturas,
                how='left',
                predicate='intersects'
            )
            
            # Calcula promedio, divide entre dos y redondea a 0 decimales
            buffer_results = buffer_join.groupby(level=0)['pt'].agg(lambda x: round(x.mean(), 0))
            
            print(f"Casos con match por buffer: {len(buffer_results)}")
            for idx in buffer_results.index:
                processed.loc[idx, 'planta'] = round(buffer_results[idx] / 2, 0)
        
        # 3. Eliminar casos sin match
        sin_match = casos_especiales.index.difference(
            matched.index.union(buffer_results.index if len(unmatched) > 0 else [])
        )
        if len(sin_match) > 0:
            print(f"Eliminando {len(sin_match)} casos sin match...")
            processed = processed.drop(index=sin_match)
        
        return processed
        
    except Exception as e:
        print(f"Error en update_floor_info: {str(e)}")
        print(f"Tipo de error: {type(e)}")
        return processed


#########  MÍNIMO PARA ASCENSOR ##########
def update_floor_info_ascensor(gdf, alturas_gdf):
    """
    Igual que el otro, pero aplica el ajuste para pisos altos sin ascensor.
    """
    print("Preparando datos...")
    # Crear copias limpias y filtrar alturas
    processed = gdf.copy()
    alturas = alturas_gdf[['geometry', 'pt']].copy()
    alturas = alturas[alturas['pt'] <= 20].copy()  # Filtrar edificios de más de 20 plantas
    
    # Reset índices
    processed = processed.reset_index(drop=True)
    alturas = alturas.reset_index(drop=True)
    
    # Identificar casos especiales según tipo de oferta
    mask_especial = (
            ((processed['ascensor'] == 0) & (processed['planta'] > 5))
        )

    casos_especiales = processed[mask_especial].copy()
    print(f"Casos especiales identificados: {len(casos_especiales)}")
    
    if len(casos_especiales) == 0:
        return processed
    
    try:
        # 1. Join directo
        print("Realizando join directo...")
        join_result = gpd.sjoin(
            casos_especiales[['geometry', 'planta']],
            alturas,
            how='left',
            predicate='within'
        )
        
        # Separar casos del join directo: solo mantener los que nueva_planta <= 5
        matched_directo = []
        matched_para_buffer = []
        
        for idx, row in join_result.iterrows():
            if pd.notnull(row['pt']):
                altura_edificio = row['pt']
                # Aplicar división solo si el resultado es >= 2
                nueva_planta = round(altura_edificio / 2, 0) if (altura_edificio / 2) >= 2 else altura_edificio
                
                if nueva_planta <= 5:
                    # Aplicar join directo
                    processed.loc[idx, 'planta'] = nueva_planta
                    matched_directo.append(idx)
                else:
                    # Pasar a buffer
                    matched_para_buffer.append(idx)
        
        print(f"Casos con match directo aplicado (nueva planta ≤ 5): {len(matched_directo)}")
        print(f"Casos de join directo que pasan a buffer (nueva planta > 5): {len(matched_para_buffer)}")
        
        # 2. Buffer para casos sin match Y casos del join directo con nueva_planta > 5
        unmatched_original = casos_especiales.index.difference(join_result[pd.notnull(join_result['pt'])].index)
        casos_para_buffer = unmatched_original.union(matched_para_buffer)
        
        if len(casos_para_buffer) > 0:
            print(f"Intentando match con buffer para {len(casos_para_buffer)} casos...")
            buffer_cases = processed.loc[casos_para_buffer].copy()
            buffer_cases.geometry = buffer_cases.geometry.buffer(300)
            
            buffer_join = gpd.sjoin(
                buffer_cases[['geometry', 'planta']],
                alturas,
                how='left',
                predicate='intersects'
            )
            
            # Calcula mínimo y aplica la misma lógica condicional
            def calcular_nueva_planta(x):
                min_altura = round(x.min(), 0)
                # Aplicar división solo si el resultado es >= 2
                return round(min_altura / 2, 0) if (min_altura / 2) >= 2 else min_altura
            
            buffer_results = buffer_join.groupby(level=0)['pt'].agg(calcular_nueva_planta)
            
            print(f"Casos con match por buffer: {len(buffer_results)}")
            for idx in buffer_results.index:
                processed.loc[idx, 'planta'] = buffer_results[idx]
        
        # 3. Eliminar casos sin match (solo los que no tuvieron match en NINGÚN método)
        todos_matched = set(matched_directo).union(set(buffer_results.index if 'buffer_results' in locals() else []))
        sin_match = casos_especiales.index.difference(todos_matched)
        
        if len(sin_match) > 0:
            print(f"Eliminando {len(sin_match)} casos sin match...")
            processed = processed.drop(index=sin_match)
        
        # 4. ELIMINAR ADICIONAL: Casos que estén en planta < 5 y sin ascensor
        mask_eliminar = (processed['ascensor'] == 0) & (processed['planta'] > 5)
        casos_eliminar_adicionales = processed[mask_eliminar].index
        if len(casos_eliminar_adicionales) > 0:
            print(f"Eliminando {len(casos_eliminar_adicionales)} casos adicionales (planta < 5 y sin ascensor)...")
            processed = processed.drop(index=casos_eliminar_adicionales)
        
        return processed
        
    except Exception as e:
        print(f"Error en update_floor_info_ascensor: {str(e)}")
        print(f"Tipo de error: {type(e)}")
        return processed


################################ UPDATE_CATASTRAL_INFO
########PROMEDIOS############

def update_catastral_info(gdf, catastro_edificios, catastro_calidad, año_dataset, buffer_distance=300):
    """
    Actualiza la información catastral (año de construcción y calidad) usando un join espacial directo
    y, si es necesario, un join con buffer. Soporta columnas 'beginning' o 'año_construccion' en el catastro.
    """

    processed = gdf.copy()

    # --- Detección de columnas disponibles ---
    posibles_año = ['beginning', 'año_construccion', 'ano_construccion', 'Año_construccion']
    año_col = next((c for c in catastro_edificios.columns if c in posibles_año), None)
    if año_col is None:
        raise KeyError(f"No se encontró ninguna columna de año de construcción en catastro_edificios. Columnas disponibles: {list(catastro_edificios.columns)}")

    posibles_calidad = ['calidad_cons', 'calidad']
    calidad_col = next((c for c in catastro_calidad.columns if c in posibles_calidad), None)
    if calidad_col is None:
        raise KeyError(f"No se encontró ninguna columna de calidad en catastro_calidad. Columnas disponibles: {list(catastro_calidad.columns)}")

    # --- Renombrar columnas para evitar conflictos de nombres en el join ---
    edificios = catastro_edificios[['geometry', año_col]].rename(columns={año_col: 'año_catastro'}).copy()
    calidad = catastro_calidad[['geometry', calidad_col]].rename(columns={calidad_col: 'calidad_catastro'}).copy()

    for df in [processed, edificios, calidad]:
        if 'index_right' in df.columns:
            df.drop(columns='index_right', inplace=True, errors='ignore')

    # --- Inicializar columnas de salida ---
    processed['año_construccion'] = None
    processed['calidad_construccion'] = None

    print(f"Registros iniciales: {len(processed)}")

    try:
        # ===== JOIN DIRECTO =====
        join_año = gpd.sjoin(processed, edificios, how='left', predicate='within')
        join_calidad = gpd.sjoin(processed, calidad, how='left', predicate='within')

        casos_con_match_año = join_año[pd.notnull(join_año['año_catastro'])]
        casos_con_match_calidad = join_calidad[pd.notnull(join_calidad['calidad_catastro'])]

        print(f"Casos con match directo (año): {len(casos_con_match_año)}")
        print(f"Casos con match directo (calidad): {len(casos_con_match_calidad)}")

        # Asignar valores directos
        processed.loc[casos_con_match_año.index, 'año_construccion'] = casos_con_match_año['año_catastro'].values
        processed.loc[casos_con_match_calidad.index, 'calidad_construccion'] = casos_con_match_calidad['calidad_catastro'].values

        # ===== JOIN CON BUFFER (para los que no tuvieron match) =====
        # --- Año ---
        casos_sin_año = processed[pd.isnull(processed['año_construccion'])].copy()
        if len(casos_sin_año) > 0:
            casos_sin_año['geometry'] = casos_sin_año.geometry.buffer(buffer_distance)
            buffer_join = gpd.sjoin(casos_sin_año, edificios, how='left', predicate='intersects')
            if not buffer_join.empty:
                promedios_año = buffer_join.groupby(buffer_join.index)['año_catastro'].mean().round(0)
                casos_con_buffer = promedios_año[pd.notnull(promedios_año)]
                print(f"Casos adicionales con match por buffer (año): {len(casos_con_buffer)}")
                processed.loc[casos_con_buffer.index, 'año_construccion'] = casos_con_buffer.values

        # --- Calidad ---
        casos_sin_calidad = processed[pd.isnull(processed['calidad_construccion'])].copy()
        if len(casos_sin_calidad) > 0:
            casos_sin_calidad['geometry'] = casos_sin_calidad.geometry.buffer(buffer_distance)
            buffer_join = gpd.sjoin(casos_sin_calidad, calidad, how='left', predicate='intersects')
            if not buffer_join.empty:
                promedios_calidad = buffer_join.groupby(buffer_join.index)['calidad_catastro'].mean().round(0)
                casos_con_buffer = promedios_calidad[pd.notnull(promedios_calidad)]
                print(f"Casos adicionales con match por buffer (calidad): {len(casos_con_buffer)}")
                processed.loc[casos_con_buffer.index, 'calidad_construccion'] = casos_con_buffer.values

        # ===== LIMPIEZA Y CÁLCULOS =====
        processed['año_construccion'] = pd.to_numeric(processed['año_construccion'], errors='coerce')
        processed['calidad_construccion'] = pd.to_numeric(processed['calidad_construccion'], errors='coerce')

        nan_count = processed['año_construccion'].isna().sum()
        if nan_count > 0:
            print(f"⚠️ {nan_count} registros sin año de construcción (ni con buffer). Se eliminarán.")
            processed = processed.dropna(subset=['año_construccion'])

        processed['año_construccion'] = processed['año_construccion'].astype(int)
        processed['antiguedad'] = (año_dataset - processed['año_construccion']).clip(lower=0)

        print(f"\nAntigüedad calculada usando año {año_dataset}")
        print(f"Registros finales tras limpieza: {len(processed)}")

        return processed

    except Exception as e:
        print(f"Error en update_catastral_info: {str(e)}")
        if 'año_construccion' in processed.columns:
            print(f"Tipos: {processed['año_construccion'].dtype}")
            print(processed[processed['año_construccion'].isna()].head())
        return processed


def encode_tipo_column(df):
    """
    Realiza one-hot encoding de la columna 'tipo' y asegura que los valores sean 0 y 1 (int).
    """
    # Crear dummies y asegurar que sean int
    dummies = pd.get_dummies(df['tipo'], prefix='tipo')
    for col in dummies.columns:
        dummies[col] = dummies[col].astype(int)
    
    # Concatenar con el DataFrame original y eliminar columna original
    df = pd.concat([df, dummies], axis=1)
    #df = df.drop(columns=['tipo'])
    
    return df

def extraer_cambio_precio(valor):
    """
    Convierte texto de 'Cambio de precio' a número NEGATIVO.
    Ejemplo:
        'ha bajado 108.000 €' → -108000
        NaN o sin número → 0
    """
    if pd.isna(valor):
        return 0

    match = re.search(r'(\d{1,3}(?:\.\d{3})*|\d+)', str(valor))
    if match:
        numero = int(match.group(1).replace('.', ''))
        return -numero  # Rebaja → número negativo
    
    return 0

def process_habitaclia_data(df, año_dataset, tipo_datos='venta', tipo_oferta='vivienda', analisis_sentimiento=False, filtro=True):
    """
    Procesa el DataFrame completo de Habitaclia aplicando todas las transformaciones necesarias.
    
    Args:
        df: DataFrame a procesar
        año_dataset: Año del dataset
        tipo_datos: 'venta' o 'alquiler' para determinar el tipo de procesamiento
        tipo_oferta: 'vivienda', 'local_comercial' u 'oficina' para determinar el tipo de oferta
        analisis_sentimiento: Si se debe realizar análisis de sentimiento
        filtros_compra: Si se deben aplicar filtros de compra
    """
    if tipo_oferta not in ['vivienda', 'local_comercial', 'oficina']:
        raise ValueError("tipo_oferta debe ser 'vivienda', 'local_comercial' u 'oficina'")
    try:
        # Copia para no modificar el original
        processed = df.copy()
        
        # 0. Extraer código de inmueble
        processed = extract_property_code(processed)
        
        # 1. Extraer información básica
        print("Extrayendo información básica...")
        basic_info = processed.apply(process_basic_info, axis=1)
        for col in basic_info.columns:
            processed[col] = basic_info[col]
        
        # 2. Procesar planta
        print("Procesando información de plantas...")
        processed['planta'] = processed.apply(
            lambda x: extract_floor_info(x['Description'], x['General Characteristics']), 
            axis=1
        )
        
        # 3. Detectar obra nueva (solo para viviendas)
        if tipo_oferta == 'vivienda':
            print("Detectando obra nueva...")
            processed['obra_nueva'] = processed.apply(has_obra_nueva, axis=1)
        else:
            processed['obra_nueva'] = 0
        
        # 4. Detectar tipo de vivienda según tipo_datos (solo para viviendas)
        if tipo_oferta == 'vivienda':
            print("Detectando tipo de vivienda...")
            if tipo_datos == 'venta':
                housing_results = processed.apply(detect_housing_type_venta, axis=1)
            elif tipo_datos == 'alquiler':
                housing_results = processed.apply(detect_housing_type_alquiler, axis=1)
            else:
                raise ValueError("tipo_datos debe ser 'venta' o 'alquiler'")
                
            processed['tipo'] = housing_results.apply(lambda x: x['tipo'])
            processed['plurifam'] = housing_results.apply(lambda x: x['plurifam'])
            
            # Identificar y ajustar casos unifamiliares a planta 0
            unifam_planta_positiva = processed[
                (processed['plurifam'] == 0) 
            ]
            print(f"Casos unifamiliares (ajustados todos a planta 0): {len(unifam_planta_positiva)}")
            processed.loc[processed['plurifam'] == 0, 'planta'] = 0
        else:
            processed['tipo'] = tipo_oferta
            processed['plurifam'] = 1
        
        # 5. Actualizar información de plantas con catastro
        print("Actualizando información de plantas con catastro...")
        if tipo_oferta == 'vivienda':
            catastro_alturas = gpd.read_file(str(DATA_DIR / 'catastro_alturas.gpkg'))
        elif tipo_oferta == 'local_comercial':
            catastro_alturas = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-locales_comerciales_con_alturas.gpkg'))
        else:  # oficina
            catastro_alturas = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-oficinas_con_alturas.gpkg'))
        processed = update_floor_info(processed, catastro_alturas)
        
        # 6. Detectar estado de ocupación (solo para viviendas)
        if tipo_oferta == 'vivienda':
            print("Detectando estado de ocupación...")
            occupancy_results = processed['Description'].apply(detect_occupancy_status)
            processed['okupa'] = occupancy_results.apply(lambda x: x['okupa'])
            processed['alquilado'] = occupancy_results.apply(lambda x: x['alquilado'])
        else:
            processed['okupa'] = 0
            processed['alquilado'] = 0
        
        # 7. Verificar cédula de habitabilidad (solo para viviendas)
        if tipo_oferta == 'vivienda':
            print("Verificando cédula de habitabilidad...")
            processed['cedula'] = processed['Description'].apply(check_habitability_certificate)
        else:
            processed['cedula'] = 1
        
        # 8. Extraer características
        print("Extrayendo características...")
        features = processed.apply(extract_property_features, axis=1)
        processed = pd.concat([processed, features], axis=1)

        # 9. Actualizar información de plantas para viviendas altas sin ascensor (solo para vivienda)
        if tipo_oferta == 'vivienda':
            processed=update_floor_info_ascensor(processed, catastro_alturas)

        # En caso de que haya columnas duplicadas, quedarnos con las nuevas
        for col in features.columns:
            if col in processed.columns and col in features.columns:
                processed[col] = features[col]
        
        # 9. Procesar EPC
        print("Procesando calificación energética...")
        epc_results = processed['EPC_Emission'].apply(process_epc_rating)
        processed = pd.concat([processed, epc_results], axis=1)
        
        # En caso de que haya columnas duplicadas, quedarnos con las nuevas
        for col in epc_results.columns:
            if col in processed.columns and col in epc_results.columns:
                processed[col] = epc_results[col]
        
        # 10. Calcular precio por metro cuadrado
        processed['precio_unitario'] = processed['precio_euros'] / processed['superficie']
        
        # 11. Actualizar información catastral
        print("Procesando datos catastrales...")
        try:
            # Seleccionar fuentes según tipo de oferta
            if tipo_oferta == 'vivienda':
                catastro_edificios = gpd.read_file(str(DATA_DIR / 'buildings_catastro.gpkg'))
                catastro_edificios = catastro_edificios[
                    catastro_edificios['currentUse'].str.contains('residential', case=False, na=False)
                ].copy()
                catastro_calidad = gpd.read_file(str(DATA_DIR / 'catastro_calidad_cons.gpkg'))
            elif tipo_oferta == 'local_comercial':
                catastro_edificios = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-locales_comerciales_con_año_construccion.gpkg'))
                catastro_calidad = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-locales_comerciales_con_calidad.gpkg'))
            else:  # oficina
                catastro_edificios = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-oficinas_con_año_construccion.gpkg'))
                catastro_calidad = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-oficinas_con_calidad.gpkg'))
            
            print(f"Registros encontrados en catastro: {len(catastro_edificios)}")
            processed = update_catastral_info(processed, catastro_edificios, catastro_calidad, año_dataset)
        except Exception as e:
            print(f"Error en procesamiento catastral: {str(e)}")
            return processed

        # 12. Procesamiento de texto y análisis de sentimiento (opcional)
        if analisis_sentimiento:
            print("\nProcesando texto y lenguaje...")
            language_results = processed['Description'].apply(process_text_language)
            processed['cl_descrip'] = language_results.apply(lambda x: x['text'])
            processed['Language'] = language_results.apply(lambda x: x['lang'])
            
            print("Analizando sentimiento...")
            sentiment_results = processed['cl_descrip'].apply(analyze_sentiment)
            processed['POS'] = sentiment_results.apply(lambda x: x['POS'])
            processed['NEU'] = sentiment_results.apply(lambda x: x['NEU'])
            processed['NEG'] = sentiment_results.apply(lambda x: x['NEG'])

        # 13: Ajustes varios (solo para viviendas)
        if tipo_oferta == 'vivienda':
            # 13.1 One-hot encoding de tipo
            print("\nRealizando one-hot encoding de tipo...")
            processed = encode_tipo_column(processed)
            # 13.2 Ajustar año_construccion y antiguedad para obra nueva
            print("\nAjustando año de construcción y antigüedad para obra nueva...")
            processed.loc[processed['obra_nueva'] == 1, 'año_construccion'] = año_dataset - 1
            processed.loc[processed['obra_nueva'] == 1, 'antiguedad'] = 1
            print(f"Ajustados {processed['obra_nueva'].sum()} casos de obra nueva")
            # 13.3 Ajustar obra_nueva según antigüedad
            print("\nAjustando obra nueva según antigüedad...")
            casos_antes = processed['obra_nueva'].sum()
            processed.loc[processed['antiguedad'] <= 1, 'obra_nueva'] = 1
            casos_despues = processed['obra_nueva'].sum()
            print(f"Ajustados {casos_despues - casos_antes} casos adicionales como obra nueva")

        #####
        # 14. Aplicar filtros (opcional) según tipo_datos y tipo_oferta
        if filtro:
            print(f"Aplicando filtros de {tipo_datos} para {tipo_oferta}...")
            print(f"Inicialmente, tenemos {len(processed)} casos")

            # Filtros básicos comunes a todos los tipos
            processed = processed[
                (processed['superficie'] > 0) &
                (processed['precio_euros'] > 0)
            ].copy()

            # Filtros específicos por tipo de oferta
            if tipo_oferta == 'vivienda':
                # --- FILTROS PARA VIVIENDAS ---
                processed = processed[processed['plurifam'] == 1].copy()
                print(f"Tras eliminar los unifamiliares, tenemos {len(processed)} casos")

                if tipo_datos == 'venta':
                    processed = processed[
                        (processed['superficie'].between(30, 300)) &
                        (processed['precio_euros'].between(5000, 20000000)) &
                        (processed['precio_unitario'].between(10, 50000))
                    ].copy()
                else:  # alquiler
                    processed = processed[
                        (processed['superficie'].between(30, 300)) &
                        (processed['precio_euros'].between(200, 20000)) &
                        (processed['precio_unitario'].between(5, 6000))
                    ].copy()

                processed = processed[
                    (~processed['tipo'].isin(['estudio', 'loft', 'atico', 'duplex', 'triplex'])) &
                    (processed['planta'] > 0)
                ].copy()

                processed = processed[processed['okupa'] == 0].copy()
                if tipo_datos == 'venta':
                    processed = processed[processed['alquilado'] == 0].copy()
                processed = processed[processed['cedula'] == 1].copy()
                processed = processed[processed['parking'].isin([0, 1])].copy()
                print(f"Tras eliminar casos con parking incluido, tenemos {len(processed)} casos")

                processed = processed[
                    (processed['n_habs'].between(1, 10)) &
                    (processed['n_baños'] <= 10) &
                    (processed['n_habs'] / processed['n_baños'] >= 0.5) &
                    (processed['superficie'] / processed['n_habs'] >= 13)
                ].copy()

            elif tipo_oferta == 'oficina':
                # --- FILTROS PARA OFICINAS ---
                if tipo_datos == 'alquiler':
                    processed = processed[
                        (processed['precio_euros'].between(200, 10000)) &
                        (processed['precio_unitario'].between(5, 130)) &
                        (processed['superficie'].between(10, 900))
                    ].copy()
                elif tipo_datos == 'venta':
                    processed = processed[
                        (processed['precio_euros'].between(50000, 1500000)) &
                        (processed['precio_unitario'].between(650, 20000)) &
                        (processed['superficie'].between(10, 900))
                    ].copy()

            elif tipo_oferta == 'local_comercial':
                # --- FILTROS PARA LOCALES ---
                if tipo_datos == 'alquiler':
                    processed = processed[
                        (processed['precio_euros'].between(200, 15000)) &
                        (processed['precio_unitario'].between(3.5, 150)) &
                        (processed['superficie'].between(10, 950))
                    ].copy()
                elif tipo_datos == 'venta':
                    processed = processed[
                        (processed['precio_euros'].between(15500, 1500000)) &
                        (processed['precio_unitario'].between(220, 25000)) &
                        (processed['superficie'].between(5, 950))
                    ].copy()

            print(f"Tras aplicar filtros de superficie y precio, tenemos {len(processed)} casos")

            # Filtros comunes a todos
            processed = processed[processed['año_construccion'] >= 1900].copy()
            print(f"Tras eliminar casos construidos antes de 1900, tenemos {len(processed)} casos")

            processed = remove_duplicates_by_code(processed)
            print(f"Tras eliminar duplicados por código de anuncio, tenemos {len(processed)} casos")
            processed = remove_duplicates_by_descrip(processed)
            print(f"Tras eliminar duplicados por descripción, tenemos {len(processed)} casos")
            processed = remove_spatial_duplicates(processed,grid_path=str(DATA_DIR / "cuadricula_bcn-31N.gpkg"),
                                                  tipo_oferta=tipo_oferta)
            print(f"Tras eliminar duplicados espaciales, tenemos {len(processed)} casos")

        else:
            print("Saltando filtros de limpieza finales (filtro=False).")

        # 15. Procesar cambio de precio
        print("\nProcesando columna 'Cambio de precio'...")
        if 'Cambio de precio' in processed.columns:
            processed['reduccion_precio'] = processed['Cambio de precio'].apply(extraer_cambio_precio)
            n_nan = processed['Cambio de precio'].isna().sum()
            print(f"Se convirtieron {len(processed) - n_nan} valores y {n_nan} estaban vacíos (se reemplazaron por 0).")
        else:
            print("No se encontró la columna 'Cambio de precio' en el DataFrame.")
    
        # 16. Calcular distancias a puntos de referencia
        print("\nCalculando distancias a puntos de referencia...")
        try:
            placa_cat = gpd.read_file(str(DATA_DIR / 'placa_catalunya_BCN_31N.gpkg'))
            fr_macia = gpd.read_file(str(DATA_DIR / 'placa_francesc_macia_BCN_31N.gpkg'))
            
            # Calcular distancias euclidianas en metros
            processed['distancia_plaza_cat'] = processed.geometry.apply(lambda x: x.distance(placa_cat.geometry.iloc[0]))
            processed['distancia_fr_macia'] = processed.geometry.apply(lambda x: x.distance(fr_macia.geometry.iloc[0]))
            print("Distancias calculadas correctamente.")
        except Exception as e:
            print(f"Error al calcular distancias: {str(e)}")
            processed['distancia_plaza_cat'] = None
            processed['distancia_fr_macia'] = None

        # 17. Calcular primera línea de mar
        print("\nCalculando primera línea de mar...")
        try:
            linea_mar = gpd.read_file(str(DATA_DIR / 'primera_linea_de_mar_BCN_31N.gpkg'))
            buffer_mar = linea_mar.geometry.buffer(300)
            processed['primera_linea_mar'] = processed.geometry.apply(
                lambda x: 1 if any(x.within(buf) for buf in buffer_mar) else 0
            )
            print(f"Primera línea de mar calculada: {processed['primera_linea_mar'].sum()} casos positivos.")
        except Exception as e:
            print(f"Error al calcular primera línea de mar: {str(e)}")
            processed['primera_linea_mar'] = 0
        
        return processed
        
    except Exception as e:
        print(f"Error en procesamiento: {str(e)}")
        return processed
    

def process_habitaclia_parking(df, año_dataset, filtro=True, tipo_datos='venta'):
    """
    Procesa datos de Habitaclia para ofertas de PARKING.
    Incluye superficie, precio unitario, reducción de precio, 
    distancias a puntos de referencia y proximidad al mar.

    Args:
        df: DataFrame o GeoDataFrame original.
        año_dataset: Año de referencia del dataset.
        filtro: Si se aplican filtros de limpieza básicos.
        tipo_datos: 'venta' o 'alquiler', define filtros específicos.
    """
    processed = df.copy()

    # 1. Extraer código de inmueble (si no existe)
    processed = extract_property_code(processed)

    # 2. Información básica
    print("Extrayendo información básica...")
    basic_info = processed.apply(process_basic_info, axis=1)
    for col in basic_info.columns:
        processed[col] = basic_info[col]

    # 3. Calcular precio unitario (€/m2)
    processed['precio_euros'] = pd.to_numeric(processed['precio_euros'], errors='coerce').fillna(0.0)
    processed['superficie']   = pd.to_numeric(processed['superficie'], errors='coerce').fillna(0.0)

    processed['precio_unitario'] = 0.0
    mask_valid = processed['superficie'] > 0
    processed.loc[mask_valid, 'precio_unitario'] = (
        (processed.loc[mask_valid, 'precio_euros'] / processed.loc[mask_valid, 'superficie']).astype(float)
    )

    # 4. Procesar cambio de precio
    print("Procesando 'Cambio de precio'...")
    if 'Cambio de precio' in processed.columns:
        processed['reduccion_precio'] = processed['Cambio de precio'].apply(extraer_cambio_precio)
    else:
        processed['reduccion_precio'] = 0

    # 5. Calcular distancias a puntos de referencia
    print("Calculando distancias a Plaza Catalunya y Francesc Macià...")
    try:
        placa_cat = gpd.read_file(str(DATA_DIR / 'placa_catalunya_BCN_31N.gpkg'))
        fr_macia  = gpd.read_file(str(DATA_DIR / 'placa_francesc_macia_BCN_31N.gpkg'))

        processed['distancia_plaza_cat'] = processed.geometry.apply(
            lambda x: x.distance(placa_cat.geometry.iloc[0]) if x and not x.is_empty else None
        )
        processed['distancia_fr_macia'] = processed.geometry.apply(
            lambda x: x.distance(fr_macia.geometry.iloc[0]) if x and not x.is_empty else None
        )
    except Exception as e:
        print(f"Error al calcular distancias: {e}")
        processed['distancia_plaza_cat'] = None
        processed['distancia_fr_macia'] = None

    # 6. Calcular si está en primera línea de mar
    print("Determinando proximidad al mar...")
    try:
        linea_mar = gpd.read_file(str(DATA_DIR / 'primera_linea_de_mar_BCN_31N.gpkg'))
        buffer_mar = linea_mar.geometry.buffer(300)
        processed['primera_linea_mar'] = processed.geometry.apply(
            lambda x: 1 if x and any(x.within(buf) for buf in buffer_mar) else 0
        )
    except Exception as e:
        print(f"Error al calcular primera línea de mar: {e}")
        processed['primera_linea_mar'] = 0

    # 7. Incorporar datos catastrales desde dos fuentes (viv_pluri y naves)
    print("\nProcesando información catastral...")

    try:
        # Parkings en viviendas plurifamiliares (`tip` inicia con 113 según catastro)
        catastro_año_viv = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-parkings_con_año_construccion-viv_pluri.gpkg'))
        catastro_calidad_viv = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-parkings_con_calidad-viv_pluri.gpkg'))

        gdf_viv = update_catastral_info(processed, catastro_año_viv, catastro_calidad_viv, año_dataset)

        processed['año_construccion-viv_pluri'] = gdf_viv.reindex(processed.index)['año_construccion']
        processed['calidad-viv_pluri'] = gdf_viv.reindex(processed.index)['calidad_construccion']

        # Parkings en naves (`tip` inicia con 22 según catastro)
        catastro_año_nav = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-parkings_con_año_construccion-naves.gpkg'))
        catastro_calidad_nav = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-parkings_con_calidad-naves.gpkg'))
        gdf_nav = update_catastral_info(processed, catastro_año_nav, catastro_calidad_nav, año_dataset)

        processed['año_construccion-naves'] = gdf_nav.reindex(processed.index)['año_construccion']
        processed['calidad-naves'] = gdf_nav.reindex(processed.index)['calidad_construccion']

        print("Datos catastrales añadidos correctamente.")
    except Exception as e:
        print(f"⚠️ Error en la carga o cruce catastral: {e}")
        processed['año_construccion-viv_pluri'] = None
        processed['calidad-viv_pluri'] = None
        processed['año_construccion-naves'] = None
        processed['calidad-naves'] = None

    # 8. Filtros básicos (opcional)
    if filtro:
        print("Aplicando filtros básicos...")
        processed = processed[
            (processed['superficie'] > 0) &
            (processed['precio_euros'] > 0) &
            (processed['precio_unitario'] > 0)
        ].copy()

        # 🔹 Filtros específicos según tipo de datos
        if tipo_datos == 'venta':
            processed = processed[
                (processed['precio_euros'] > 1000) &
                (processed['superficie'].between(10, 70))
            ].copy()
        elif tipo_datos == 'alquiler':
            processed = processed[
                (processed['precio_euros'].between(20, 400)) &
                (processed['superficie'].between(10, 70))
            ].copy()
        else:
            print(f"⚠️ tipo_datos '{tipo_datos}' no reconocido. No se aplican filtros específicos.")

    # 9. Eliminar duplicados
    processed = remove_duplicates_by_code(processed.copy())
    print(f"Tras eliminar duplicados por código de anuncio: {len(processed)} registros.")
    processed = processed.drop_duplicates(subset=['Description'], keep='first')
    print(f"Tras eliminar duplicados por descripción: {len(processed)} registros.")
    processed = remove_spatial_duplicates(processed,grid_path=str(DATA_DIR / "cuadricula_bcn-31N.gpkg"), tipo_oferta="parking")
    print(f"Tras eliminar duplicados espaciales, tenemos {len(processed)} casos")

    print(f"Procesamiento completado. Total de parkings: {len(processed)} registros.")
    return processed


def process_habitaclia_industrial(df, año_dataset):
    """
    Procesa datos de Habitaclia para ofertas de industrial.
    Incluye superficie, precio unitario, reducción de precio, 
    distancias a puntos de referencia y proximidad al mar.

    Args:
        df: DataFrame o GeoDataFrame original.
        año_dataset: Año de referencia del dataset.
    """
    processed = df.copy()

    # 1. Extraer código de inmueble (si no existe)
    if 'codigo_inmueble' not in processed.columns and 'Link' in processed.columns:
        processed = extract_property_code(processed)

    # 2. Información básica
    print("Extrayendo información básica...")
    basic_info = processed.apply(process_basic_info, axis=1)
    for col in basic_info.columns:
        processed[col] = basic_info[col]

    # 3. Calcular precio unitario (€/m2)
    processed['precio_euros'] = pd.to_numeric(processed['precio_euros'], errors='coerce').fillna(0.0)
    processed['superficie']   = pd.to_numeric(processed['superficie'], errors='coerce').fillna(0.0)

    processed['precio_unitario'] = 0.0
    mask_valid = processed['superficie'] > 0
    processed.loc[mask_valid, 'precio_unitario'] = (
        (processed.loc[mask_valid, 'precio_euros'] / processed.loc[mask_valid, 'superficie']).astype(float)
    )

    # 4. Procesar cambio de precio
    print("Procesando 'Cambio de precio'...")
    if 'Cambio de precio' in processed.columns:
        processed['reduccion_precio'] = processed['Cambio de precio'].apply(extraer_cambio_precio)
    else:
        processed['reduccion_precio'] = 0

    # 7. Incorporar datos catastrales
    print("\nProcesando información catastral...")

    catastro_año = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-locales_industriales_con_año_construccion.gpkg'))
    catastro_calidad = gpd.read_file(str(DATA_DIR / '04-parcelas_catastrales-locales_industriales_con_calidad.gpkg'))

    gdf = update_catastral_info(processed, catastro_año, catastro_calidad, año_dataset, buffer_distance=1200)

    processed['año_construccion'] = gdf.reindex(processed.index)['año_construccion']
    processed['calidad'] = gdf.reindex(processed.index)['calidad_construccion']

    # 8. Filtros básicos (opcional)
    print("Aplicando filtros básicos...")
    
    processed = processed[
            (processed['superficie'] > 0) &
            (processed['precio_euros'] > 0) &
            (processed['precio_unitario'] > 0)
        ].copy()

    # 9. Eliminar duplicados
    processed = remove_duplicates_by_code(processed.copy())
    print(f"Tras eliminar duplicados por código de anuncio: {len(processed)} registros.")
    processed = processed.drop_duplicates(subset=['Description'], keep='first')
    print(f"Tras eliminar duplicados por descripción: {len(processed)} registros.")
    processed = remove_spatial_duplicates(processed,grid_path=str(DATA_DIR / "cuadricula_bcn-31N.gpkg"), tipo_oferta="industrial")
    print(f"Tras eliminar duplicados espaciales, tenemos {len(processed)} casos")

    print(f"Procesamiento completado. Total de industrials: {len(processed)} registros.")
    return processed