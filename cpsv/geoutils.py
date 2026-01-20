"""
Utilidades geoespaciales pequeñas y seguras para los notebooks.

Objetivo: centralizar operaciones repetidas que aparecen en los notebooks
sin cambiar su comportamiento (weighted intersections, conteo de paradas,
añadir áreas de parques, creación de buffers).

Estas funciones usan geopandas y pandas y preservan columnas originales.
"""
from typing import List, Optional
import geopandas as gpd
import pandas as pd
# from pathlib import Path
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
import numpy as np
import osmnx as ox
from fuzzywuzzy import fuzz
from unidecode import unidecode

def weighted_intersection(buffers: gpd.GeoDataFrame,
                          census_tracts: gpd.GeoDataFrame,
                          weighted_cols: List[str],
                          buffer_id_col: str = 'buffer_id',
                          normalize_by_covered: bool = True) -> gpd.GeoDataFrame:
    """
    Realiza la intersección entre `buffers` y `census_tracts`, calcula la
    proporción de área de cada intersección respecto al buffer original y
    devuelve un GeoDataFrame con las columnas ponderadas (suma por buffer).

    Si `normalize_by_covered` es True (por defecto), cuando las secciones
    censales no cubren el 100% del área del buffer, la suma ponderada se
    normaliza por la fracción cubierta (es decir, la parte cubierta se
    considera como 100%). Si es False, la parte no cubierta contribuye como 0
    (comportamiento histórico).

    Args:
        buffers: GeoDataFrame con geometría de buffers (se mantendrán sus columnas).
        census_tracts: GeoDataFrame con las variables a ponderar.
        weighted_cols: lista de nombres de columnas en `census_tracts` a ponderar.
        buffer_id_col: nombre de la columna temporal usada internamente.
        normalize_by_covered: si True normaliza la suma ponderada por la fracción
                              del buffer efectivamente cubierta por los tramos.

    Returns:
        GeoDataFrame con geometría de los buffers y columnas ponderadas.
    """
    # Copiar para no modificar los originales
    buffers = buffers.copy()
    census_tracts = census_tracts.copy()

    # Asegurar un índice consecutivo en buffers para mapear resultados
    buffers = buffers.reset_index(drop=True)
    buffers[buffer_id_col] = range(len(buffers))

    # Guardar columnas originales (excepto geometry)
    original_columns = [c for c in buffers.columns if c not in ['geometry', buffer_id_col]]

    # Precalcular áreas de buffer (evita llamadas repetidas a .area dentro de apply)
    # Usamos una Serie para permitir un mapeo vectorizado desde la intersección.
    buffer_areas = pd.Series(buffers.geometry.area.values, index=buffers[buffer_id_col].values)

    # Intersección
    intersection = gpd.overlay(buffers, census_tracts, how='intersection')
    if intersection.empty:
        # Crear un resultado vacío con las columnas solicitadas
        result = buffers.copy()
        for col in weighted_cols:
            result[col] = 0
        # Mantener geometría y devolver
        return gpd.GeoDataFrame(result, geometry=result.geometry, crs=buffers.crs)

    # Calcular proporción de área respecto al buffer (vectorizado)
    # Mapear el área del buffer correspondiente a cada fila de la intersección
    intersection['buffer_area'] = intersection[buffer_id_col].map(buffer_areas).fillna(1)
    intersection['area_proportion'] = intersection.geometry.area / intersection['buffer_area']

    # Calcular columnas ponderadas
    for col in weighted_cols:
        if col in intersection.columns:
            intersection[f'weighted_{col}'] = intersection[col] * intersection['area_proportion']
        else:
            # Si la columna no existe en census_tracts, crear 0 para evitar errores
            intersection[f'weighted_{col}'] = 0

    # Agrupar por buffer_id y sumar las columnas ponderadas
    agg_dict = {f'weighted_{col}': 'sum' for col in weighted_cols}
    # Mantener la primera ocurrencia de las columnas originales del buffer
    agg_dict.update({col: 'first' for col in original_columns})

    grouped = intersection.groupby(buffer_id_col).agg(agg_dict)

    # Si se desea normalizar por la fracción cubierta, calcularla y ajustar
    if normalize_by_covered:
        covered_frac = intersection.groupby(buffer_id_col)['area_proportion'].sum()
        # Evitar división por 0: solo normalizamos cuando covered_frac > 0
        for col in weighted_cols:
            wcol = f'weighted_{col}'
            if wcol in grouped.columns:
                # dividir y mantener NaN para buffers no cubiertos (se rellenarán luego)
                grouped[wcol] = grouped[wcol].div(covered_frac).replace([pd.NA, float('inf')], pd.NA)

    # Renombrar las columnas para quitar el prefijo weighted_
    rename_map = {f'weighted_{col}': col for col in weighted_cols}
    grouped = grouped.rename(columns=rename_map)

    # Construir GeoDataFrame final reusando la geometría original del buffer
    geom = buffers.set_index(buffer_id_col).geometry
    result = gpd.GeoDataFrame(grouped, geometry=geom, crs=buffers.crs)

    # Asegurar que todas las columnas solicitadas existan en el resultado
    for col in weighted_cols:
        if col not in result.columns:
            result[col] = 0
    # Para aquellos buffers sin cobertura, rellenar con 0 (comportamiento razonable)
    for col in weighted_cols:
        result[col] = result[col].fillna(0)

    # Reordenar columnas: mantener original_columns primero si están
    final_cols = [c for c in original_columns if c in result.columns] + [c for c in weighted_cols if c in result.columns]
    # Evitar KeyError
    try:
        result = result[final_cols + ['geometry']]
    except Exception:
        # Si algo falla con el reordenamiento, devolver tal cual
        pass

    return result.reset_index(drop=True)

def get_metropolitan_parks(municipalities: Optional[List[str]] = None) -> gpd.GeoDataFrame:
    if municipalities is None:
        municipalities = [
            'Barcelona, Spain',
            'Hospitalet de Llobregat, Spain', 
            'Santa Coloma de Gramenet, Spain',
            'Badalona, Spain',
            'Sant Adria de Besos, Spain',
            'Montcada i Reixac, Spain',
            'Cerdanyola del Vallès, Spain',
            'Sant Cugat del Vallès, Spain',
            'Molins de Rei, Spain',
            'Sant Feliu de Llobregat, Spain',
            'Sant Just Desvern, Spain',
            'Esplugues de Llobregat, Spain',  # corregido
        ]

    gdfs = []
    for m in municipalities:
        try:
            gdf = ox.geocode_to_gdf(m)
            gdfs.append(gdf)
        except Exception as e:
            print(f"No se pudo geocodificar '{m}': {e}")

    metropolitan_area = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    parks_tags = {'leisure': 'park'}
    parks_gdf = ox.features_from_polygon(
        metropolitan_area.unary_union,
        tags=parks_tags
    )

    if not parks_gdf.empty:
        parks_gdf = parks_gdf[parks_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        parks_gdf = parks_gdf[['geometry', 'name']].reset_index(drop=True)
        if parks_gdf.crs:
            parks_gdf = parks_gdf.to_crs('EPSG:4326')

    return parks_gdf

def count_transit_stops(buffers_gdf: gpd.GeoDataFrame, stops_gdf: gpd.GeoDataFrame,
                        buffer_index_name: Optional[str] = None) -> gpd.GeoDataFrame:
    """
    Cuenta paradas de transporte público (puntos) dentro de cada buffer.

    Devuelve una copia de `buffers_gdf` con una columna `num_tp` con el conteo.
    """
    # Reset índices para garantizar index_right coherente
    buffers = buffers_gdf.reset_index(drop=True).copy()
    stops = stops_gdf.reset_index(drop=True).copy()

    joined = gpd.sjoin(stops, buffers, predicate='within', how='left')
    if joined.empty:
        result = buffers.copy()
        result['num_tp'] = 0
        return result

    stops_count = joined.groupby('index_right').size()

    result = buffers.copy()
    # Alineamos el índice del conteo con el rango de buffers
    result['num_tp'] = stops_count.reindex(range(len(result))).fillna(0).astype(int).values
    return result


def create_buffers_from_points(gdf: gpd.GeoDataFrame, distance_m: float) -> gpd.GeoDataFrame:
    """Crea buffers en metros a partir de un GeoDataFrame en CRS proyectada (metros).
    No transforma CRS dentro de la función: el llamador debe asegurarse del CRS.
    """
    df = gdf.copy()
    df['geometry'] = df.geometry.buffer(distance_m)
    return df

def add_park_areas(buffers_gdf, parks_gdf):
    # Realizar la unión espacial
    joined = gpd.sjoin(buffers_gdf, parks_gdf[['area_m2', 'geometry']], how='left', predicate='intersects')
    
    # Eliminar 'index_right' si existe
    if 'index_right' in joined.columns:
        joined = joined.drop(columns='index_right')

    # Sumar áreas de parques por cada 'codigo_inmueble'
    park_areas = joined.groupby('codigo_inmueble')['area_m2'].sum().fillna(0)

    # Unir resultados al GeoDataFrame original
    result = buffers_gdf.copy()
    result = result.merge(park_areas, on='codigo_inmueble', how='left')

    # Crear columna final
    result['area_parques_m2'] = result['area_m2'].fillna(0)
    result = result.drop(columns=['area_m2'])
    
    return result

def buffers_to_points(buffers_gdf, keep_columns=True):
    """
    Convierte un GeoDataFrame de buffers (polígonos) en puntos,
    tomando el centroide de cada geometría.

    Args:
        buffers_gdf (gpd.GeoDataFrame): GeoDataFrame con geometrías tipo polígono.
        keep_columns (bool): Si True, conserva las columnas originales.
                             Si False, devuelve solo las columnas geométricas y el índice.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame con geometrías tipo Point (centroides).
    """
    # Copiar para no modificar el original
    gdf = buffers_gdf.copy()

    # Calcular centroides (vectorizado y eficiente)
    centroids = gdf.geometry.centroid

    # Crear GeoDataFrame de salida
    if keep_columns:
        result = gdf.drop(columns='geometry', errors='ignore').copy()
        result['geometry'] = centroids
    else:
        result = gpd.GeoDataFrame(geometry=centroids)

    # Asegurar CRS
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=gdf.crs)

    return result


##### ---- Para 01-parques.ipynb ---- ###

def clean_and_standardize_name(name: str) -> str:
    """
    Estandariza nombres para mejor comparación:
    - Convierte a minúsculas
    - Elimina acentos y caracteres especiales
    - Reemplaza palabras comunes en catalán y castellano
    """
    if pd.isna(name):
        return ""

    name = str(name).lower()
    name = unidecode(name)

    replacements = {
        'parc': 'park',
        'placa': 'plaza',
        'plaça': 'plaza',
        'placa de': 'plaza',
        'plaça de': 'plaza',
        'plaza de': 'plaza',
        'jardins': 'jardines',
        'jardí': 'jardin',
        'jardi': 'jardin',
        'jardines de': 'jardines',
        'jardin de': 'jardin',
        'park de': 'park',
        'dels': 'de los',
        'del': 'del',
        'de la': 'de la',
        'de les': 'de las',
        'de n': 'de',
        'de s': 'de',
        'de l': 'de',
        ' i ': ' y ',
        'sant': 'san',
        'santa': 'santa',
        'gran via': 'granvia',
        'gran via de': 'granvia',
        'passeig': 'paseo',
        'carrer': 'calle',
        'avinguda': 'avenida',
        'zona verda': 'zona verde'
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return name

### ---- Para 01-transporte.ipynb ---- ###

def get_metropolitan_transport():
    """
    Descarga ubicaciones de estaciones de metro, paradas de bus y tranvía en Barcelona y municipios adyacentes
    """
    # Definir áreas
    municipalities = [
        'Barcelona, Spain',
        'Hospitalet de Llobregat, Spain',
        'Santa Coloma de Gramenet, Spain',
        'Badalona, Spain',
        'Sant Adria de Besos, Spain',
        'Montcada i Reixac, Spain',
        'Cerdanyola del Vallès, Spain',
        'Sant Cugat del Vallès, Spain',
        'Molins de Rei, Spain',
        'Sant Feliu de Llobregat, Spain',
        'Sant Just Desvern, Spain',
        'Esplugues de Llobregat, Spain'
    ]
    
    # Obtener y unir geometrías de municipios
    gdfs = [ox.geocode_to_gdf(municipality) for municipality in municipalities]
    metropolitan_area = gpd.GeoDataFrame(pd.concat(gdfs))
    
    # Estaciones de metro
    metro_tags = {'railway': 'station', 'station': 'subway'}
    metro_gdf = ox.features_from_polygon(
        metropolitan_area.unary_union,
        tags=metro_tags
    )
    
    # Paradas de bus
    bus_tags = {'highway': 'bus_stop'}
    bus_gdf = ox.features_from_polygon(
        metropolitan_area.unary_union,
        tags=bus_tags
    )
    
    # Paradas de tranvía
    tram_tags = {'railway': 'tram_stop'}
    tram_gdf = ox.features_from_polygon(
        metropolitan_area.unary_union,
        tags=tram_tags
    )
    
    # Limpiar y guardar
    if not metro_gdf.empty:
        metro_gdf = metro_gdf[metro_gdf.geometry.type == 'Point']
        columns_to_keep = ['geometry', 'name']
        if 'ref' in metro_gdf.columns:
            columns_to_keep.append('ref')
        metro_gdf = metro_gdf[columns_to_keep].reset_index(drop=True)
        # Eliminar duplicados basándose en el nombre y la geometría
        metro_gdf = metro_gdf.drop_duplicates(subset=['name'])
        
    if not bus_gdf.empty:
        bus_gdf = bus_gdf[bus_gdf.geometry.type == 'Point']
        bus_gdf = bus_gdf[['geometry', 'name']].reset_index(drop=True)
        # Eliminar duplicados basándose en el nombre y la geometría
        bus_gdf = bus_gdf.drop_duplicates(subset=['name'])
        
    if not tram_gdf.empty:
        tram_gdf = tram_gdf[tram_gdf.geometry.type == 'Point']
        tram_gdf = tram_gdf[['geometry', 'name']].reset_index(drop=True)
        # Eliminar duplicados basándose en el nombre y la geometría
        tram_gdf = tram_gdf.drop_duplicates(subset=['name'])
        
    return metro_gdf, bus_gdf, tram_gdf


### ---- Para 05-control_espacial-locales_comerciales_2025.ipynb ---- ###

# -------------------------------------------------------------------
# FUNCIÓN: Asignar a cada parcela a pie de calle la línea de calle
# con la que comparte mayor longitud de borde
# -------------------------------------------------------------------
def asignar_linea_a_parcela(parcelas_gdf, segmentos_gdf):
    """
    Asigna a cada parcela en contacto con la calle (pie_de_calle=1)
    la línea de calle con la mayor longitud de intersección.
    """
    parcelas_calle = parcelas_gdf[parcelas_gdf['pie_de_calle'] == 1].copy()
    parcelas_calle['linea_id'] = None

    segmentos_sindex = segmentos_gdf.sindex

    for idx, parcela in parcelas_calle.iterrows():
        poss_idx = list(segmentos_sindex.intersection(parcela.geometry.bounds))
        posibles = segmentos_gdf.iloc[poss_idx]

        max_long = 0
        mejor_linea = None

        for _, seg in posibles.iterrows():
            seg_buffer = seg.geometry.buffer(1.5)

            if parcela.geometry.intersects(seg_buffer):
                inter = parcela.geometry.boundary.intersection(seg_buffer)

                if not inter.is_empty:
                    long_inter = inter.length
                    if long_inter > max_long:
                        max_long = long_inter
                        mejor_linea = seg['linea_id']

        if mejor_linea is not None:
            parcelas_calle.at[idx, 'linea_id'] = mejor_linea

    parcelas_gdf.loc[parcelas_calle.index, 'linea_id'] = parcelas_calle['linea_id']
    return parcelas_gdf

# -------------------------------------------------------------------
# FUNCIÓN: Asignar línea a parcelas que no recibieron ninguna
# usando la parcela colindante más grande que sí tenga línea
# -------------------------------------------------------------------
def asignar_linea_a_parcelas_null(parcelas_gdf):
    """
    Completa la asignación de línea_id copiando el valor de la
    parcela colindante (con mayor área) que ya tenga línea.
    """
    parcelas_sin = parcelas_gdf[
        (parcelas_gdf['pie_de_calle'] == 1) &
        (parcelas_gdf['linea_id'].isna())
    ].copy()

    if len(parcelas_sin) == 0:
        return parcelas_gdf

    parcelas_con = parcelas_gdf[parcelas_gdf['linea_id'].notna()]
    sindex = parcelas_con.sindex
    asignaciones = {}

    for idx, parcela in parcelas_sin.iterrows():
        buf = parcela.geometry.buffer(1)
        poss_idx = list(sindex.intersection(buf.bounds))
        posibles = parcelas_con.iloc[poss_idx]
        colindantes = posibles[posibles.intersects(buf)]

        if len(colindantes) > 0:
            mayor = colindantes.loc[colindantes.geometry.area.idxmax()]
            asignaciones[idx] = mayor['linea_id']

    for idx, linea_id in asignaciones.items():
        parcelas_gdf.at[idx, 'linea_id'] = linea_id

    return parcelas_gdf

# -------------------------------------------------------------------
# FUNCIÓN: Fusionar frentes que tengan solo una parcela
# uniéndolos con el frente colindante más grande
# -------------------------------------------------------------------
def fusionar_frentes_unitarios(frentes_calle, parcelas_gdf):
    """
    Identifica frentes compuestos por una única parcela y los fusiona
    con el frente contiguo de mayor superficie.
    """
    parcelas_por_frente = parcelas_gdf[
        parcelas_gdf['linea_id'].notna()
    ].groupby(
        parcelas_gdf['manzana_id'].astype(str) + '_' + parcelas_gdf['linea_id'].astype(str)
    ).size()

    frentes_unit = parcelas_por_frente[parcelas_por_frente == 1].index.tolist()

    if len(frentes_unit) == 0:
        return frentes_calle, parcelas_gdf

    frentes_calle = frentes_calle.copy()
    frentes_elim = set()
    fusiones = {}

    sindex = frentes_calle.sindex

    for frente_id in frentes_unit:
        if frente_id in frentes_elim:
            continue

        geom = frentes_calle[frentes_calle['frente_id'] == frente_id].geometry.iloc[0]
        buf = geom.buffer(1)

        poss_idx = list(sindex.intersection(buf.bounds))
        posibles = frentes_calle.iloc[poss_idx]

        colind = posibles[
            (posibles['frente_id'] != frente_id) &
            (posibles.intersects(buf))
        ]

        if len(colind) > 0:
            mayor = colind.loc[colind.geometry.area.idxmax()]
            fusiones[frente_id] = mayor['frente_id']
            frentes_elim.add(frente_id)

    for origen, destino in fusiones.items():
        geom_origen = frentes_calle[frentes_calle['frente_id'] == origen].geometry.iloc[0]
        idx_dest = frentes_calle[frentes_calle['frente_id'] == destino].index[0]
        geom_dest = frentes_calle.at[idx_dest, 'geometry']

        frentes_calle.at[idx_dest, 'geometry'] = geom_dest.union(geom_origen)

        parcelas_gdf.loc[
            parcelas_gdf['manzana_id'].astype(str) + '_' + parcelas_gdf['linea_id'].astype(str) == origen,
            'linea_id'
        ] = int(destino.split('_')[1])

    frentes_calle = frentes_calle[~frentes_calle['frente_id'].isin(frentes_elim)].reset_index(drop=True)

    return frentes_calle, parcelas_gdf

# -------------------------------------------------------------------
# FUNCIÓN: Convertir perímetros en segmentos simples
# -------------------------------------------------------------------
def segmentar_perimetro(manzanas_gdf, col_id=None):
    """
    Segmenta el perímetro de cada manzana en líneas simples entre vértices consecutivos.
    """
    segmentos = []

    for idx, row in manzanas_gdf.iterrows():
        manzana_id = row[col_id] if col_id and col_id in row.index else idx
        boundary = row.geometry.boundary

        if boundary.geom_type == 'MultiLineString':
            boundary = linemerge(boundary)

        if boundary.geom_type == 'MultiLineString':
            for line in boundary.geoms:
                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    linea = LineString([coords[i], coords[i+1]])
                    segmentos.append({
                        'geometry': linea,
                        'manzana_id': manzana_id,
                        'longitud': linea.length
                    })
        else:
            coords = list(boundary.coords)
            for i in range(len(coords) - 1):
                linea = LineString([coords[i], coords[i+1]])
                segmentos.append({
                    'geometry': linea,
                    'manzana_id': manzana_id,
                    'longitud': linea.length
                })

    return gpd.GeoDataFrame(segmentos, crs=manzanas_gdf.crs)

# -------------------------------------------------------------------
# FUNCIÓN: Agrupar segmentos consecutivos con orientación similar
# -------------------------------------------------------------------
def agrupar_segmentos_colineales_optimizado(segmentos_gdf, tolerancia_angular=5):
    """
    Agrupa segmentos consecutivos cuyo ángulo sea similar, formando líneas de calle.
    """
    lineas = []

    for manzana_id in segmentos_gdf['manzana_id'].unique():
        segs = segmentos_gdf[segmentos_gdf['manzana_id'] == manzana_id].copy()

        if len(segs) == 0:
            continue

        segs_ord = ordenar_segmentos_perimetro_optimizado(segs)
        segs_ord['angulo'] = segs_ord.geometry.apply(calcular_angulo)

        grupos = []
        grupo = [0]

        for i in range(1, len(segs_ord)):
            ang_act = segs_ord.iloc[i]['angulo']
            ang_prev = segs_ord.iloc[grupo[-1]]['angulo']
            diff = abs(ang_act - ang_prev)
            diff = min(diff, 360 - diff)

            if diff <= tolerancia_angular:
                grupo.append(i)
            else:
                grupos.append(grupo)
                grupo = [i]

        grupos.append(grupo)

        for g in grupos:
            geoms = list(segs_ord.iloc[g].geometry)
            fusion = linemerge(geoms)

            if fusion.geom_type == 'MultiLineString':
                fusion = max(fusion.geoms, key=lambda x: x.length)

            lineas.append({
                'geometry': fusion,
                'manzana_id': manzana_id,
                'longitud': fusion.length,
                'num_segmentos': len(g)
            })

    return gpd.GeoDataFrame(lineas, crs=segmentos_gdf.crs)

# -------------------------------------------------------------------
def ordenar_segmentos_perimetro_optimizado(segmentos):
    """
    Ordena los segmentos de un perímetro siguiendo su continuidad espacial.
    """
    if len(segmentos) <= 1:
        return segmentos

    segmentos = segmentos.reset_index(drop=True)
    orden = []
    usados = set()

    idx = 0
    orden.append(segmentos.iloc[idx])
    usados.add(idx)

    ultimo = Point(list(segmentos.iloc[idx].geometry.coords)[-1])
    sindex = segmentos.sindex

    while len(usados) < len(segmentos):
        cand_idx = list(sindex.intersection(ultimo.buffer(0.1).bounds))
        cand_idx = [i for i in cand_idx if i not in usados]

        if not cand_idx:
            distancias = []
            for i in range(len(segmentos)):
                if i in usados:
                    continue
                coords = list(segmentos.iloc[i].geometry.coords)
                d1 = ultimo.distance(Point(coords[0]))
                d2 = ultimo.distance(Point(coords[-1]))
                distancias.append((i, min(d1, d2)))

            if not distancias:
                break

            siguiente = min(distancias, key=lambda x: x[1])[0]
        else:
            siguiente = None
            dmin = float('inf')
            for i in cand_idx:
                coords = list(segmentos.iloc[i].geometry.coords)
                d1 = ultimo.distance(Point(coords[0]))
                d2 = ultimo.distance(Point(coords[-1]))
                d = min(d1, d2)
                if d < dmin:
                    dmin = d
                    siguiente = i

        if siguiente is None:
            break

        coords = list(segmentos.iloc[siguiente].geometry.coords)
        d1 = ultimo.distance(Point(coords[0]))
        d2 = ultimo.distance(Point(coords[-1]))

        seg = segmentos.iloc[siguiente].copy()
        if d2 < d1:
            seg.geometry = LineString(coords[::-1])
            ultimo = Point(coords[0])
        else:
            ultimo = Point(coords[-1])

        orden.append(seg)
        usados.add(siguiente)

    return gpd.GeoDataFrame(orden, crs=segmentos.crs)

# -------------------------------------------------------------------
def calcular_angulo(linea):
    """Calcula el ángulo de un segmento en grados."""
    coords = list(linea.coords)
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    return np.degrees(np.arctan2(dy, dx)) % 360

# -------------------------------------------------------------------
# FUNCIÓN: Obtención de locales OSM
# -------------------------------------------------------------------

CATEGORY_MAPPING = {
    # ==================== SALUD Y BIENESTAR ====================
    "doctors": "health_general",
    "doctor": "health_general",
    "clinic": "health_general",
    "dentist": "health_dental",
    "pharmacy": "health_pharmacy",
    "physiotherapist": "health_specialist",
    "midwife": "health_specialist",
    "laboratory": "health_lab",
    "optometrist": "health_specialist",
    "optician": "health_specialist",
    "massage": "health_wellness",
    "medical": "health_general",
    "medical_supply": "health_pharmacy",
    "orthopedics": "health_specialist",
    "hearing_aids": "health_specialist",
    "chemist": "health_pharmacy",
    
    # ==================== FINANZAS ====================
    "bank": "finance",
    "atm": "finance",
    "bureau_de_change": "finance",
    "money_lender": "finance",
    "pawnbroker": "finance",
    "gold_buyer": "finance",
    
    # ==================== ALIMENTACIÓN - FRESCOS ====================
    "supermarket": "food_supermarket",
    "convenience": "food_convenience",
    "bakery": "food_bakery",
    "pastry": "food_bakery",
    "butcher": "food_fresh",
    "butcher;cheese": "food_fresh",
    "greengrocer": "food_fresh",
    "greengrocer;health_food": "food_fresh",
    "seafood": "food_fresh",
    "cheese": "food_fresh",
    "dairy": "food_fresh",
    "farm": "food_fresh",
    "frozen_food": "food_fresh",
    
    # ==================== ALIMENTACIÓN - ESPECIALIZADA ====================
    "alcohol": "food_specialty",
    "beverages": "food_specialty",
    "wine": "food_specialty",
    "coffee": "food_specialty",
    "tea": "food_specialty",
    "chocolate": "food_specialty",
    "confectionery": "food_specialty",
    "confectionery;ice_cream": "food_specialty",
    "ice_cream": "food_specialty",
    "pasta": "food_specialty",
    "nuts": "food_specialty",
    "health_food": "food_specialty",
    "deli": "food_specialty",
    "charcutier;cheese;wine;butcher": "food_fresh",
    
    # ==================== ALIMENTACIÓN - GENERAL ====================
    "food": "food_general",
    "grocery": "food_general",
    "kiosk": "food_general",
    "newsagent": "retail_misc",
    
    # ==================== MODA Y ACCESORIOS ====================
    "clothes": "fashion_clothing",
    "shoes": "fashion_clothing",
    "boutique": "fashion_clothing",
    "fashion_accessories": "fashion_accessories",
    "accessories": "fashion_accessories",
    "bag": "fashion_accessories",
    "watches": "fashion_accessories",
    "jewelry": "fashion_accessories",
    "leather": "fashion_accessories",
    
    # ==================== HOGAR - MUEBLES Y DECORACIÓN ====================
    "furniture": "home_furniture",
    "bed": "home_furniture",
    "kitchen": "home_furniture",
    "bathroom_furnishing": "home_furniture",
    "interior_decoration": "home_decor",
    "curtain": "home_decor",
    "carpet": "home_decor",
    "lighting": "home_decor",
    "houseware": "home_decor",
    "tableware": "home_decor",
    "household_linen": "home_decor",
    "candles": "home_decor",
    
    # ==================== HOGAR - BRICOLAJE Y CONSTRUCCIÓN ====================
    "hardware": "home_hardware",
    "doityourself": "home_hardware",
    "paint": "home_hardware",
    "flooring": "home_hardware",
    "doors": "home_hardware",
    "stairs": "home_hardware",
    "locksmith": "home_hardware",
    "tool_hire": "home_hardware",
    "electric_supplies": "home_hardware",
    "electrical": "home_hardware",
    "hvac": "home_hardware",
    
    # ==================== HOGAR - ELECTRODOMÉSTICOS ====================
    "appliance": "home_appliances",
    "vacuum_cleaner": "home_appliances",
    
    # ==================== TECNOLOGÍA ====================
    "electronics": "tech_electronics",
    "computer": "tech_electronics",
    "mobile_phone": "tech_mobile",
    "mobile_phone_accessories": "tech_mobile",
    "telecommunication": "tech_mobile",
    "hifi": "tech_electronics",
    "radiotechnics": "tech_electronics",
    "camera": "tech_electronics",
    "printer_ink": "tech_electronics",
    
    # ==================== CULTURA Y OCIO ====================
    "books": "culture_books",
    "copyshop;books": "culture_books",
    "stationery": "culture_stationery",
    "copyshop": "culture_stationery",
    "printshop": "culture_stationery",
    "printing": "culture_stationery",
    "art": "culture_art",
    "frame": "culture_art",
    "music": "culture_media",
    "musical_instrument": "culture_media",
    "video": "culture_media",
    "video_games": "culture_media",
    "games": "culture_media",
    "toys": "culture_toys",
    "model": "culture_toys",
    "collector": "culture_toys",
    
    # ==================== SERVICIOS PERSONALES ====================
    "hairdresser": "service_beauty",
    "beauty": "service_beauty",
    "cosmetics": "service_beauty",
    "perfumery": "service_beauty",
    "hairdresser_supply": "service_beauty",
    "tattoo": "service_beauty",
    "laundry": "service_cleaning",
    "dry_cleaning": "service_cleaning",
    "tailor": "service_clothing_repair",
    "sewing": "service_clothing_repair",
    "shoe_repair": "service_clothing_repair",
    "watch_repair": "service_repair",
    "bicycle_repair": "service_repair",
    "car_repair": "service_automotive",
    "car;car_repair": "service_automotive",
    "motorcycle_repair": "service_automotive",
    "repair": "service_repair",
    
    # ==================== VEHÍCULOS ====================
    "car": "vehicle_car",
    "car_parts": "vehicle_car",
    "tyres": "vehicle_car",
    "bicycle": "vehicle_bicycle",
    "bikes": "vehicle_bicycle",
    "motorcycle": "vehicle_motorcycle",
    "motorbike rent and travels": "vehicle_motorcycle",
    "electric_scooter": "vehicle_other",
    "mobility_scooter": "vehicle_other",
    
    # ==================== DEPORTE Y EXTERIOR ====================
    "sports": "sport_general",
    "outdoor": "sport_general",
    "fishing": "sport_general",
    "skate": "sport_general",
    "surf;skate;snowboard": "sport_general",
    "garden_centre": "home_garden",
    "agrarian": "home_garden",
    
    # ==================== MASCOTAS ====================
    "pet": "pet",
    "pet_grooming": "pet",
    "dog_wash": "pet",
    
    # ==================== PROFESIONALES Y SERVICIOS ====================
    "estate_agent": "service_professional",
    "lawyer": "service_professional",
    "travel_agency": "service_professional",
    "ticket": "service_professional",
    "bus_tickets": "service_professional",
    "training_centre": "service_professional",
    
    # ==================== RETAIL GENERAL ====================
    "department_store": "retail_department",
    "mall": "retail_department",
    "variety_store": "retail_general",
    "general": "retail_general",
    "gift": "retail_general",
    "second_hand": "retail_general",
    "charity": "retail_general",
    "antiques": "retail_general",
    "florist": "retail_general",
    "party": "retail_general",
    
    # ==================== OTROS ESPECIALIZADOS ====================
    "bookmaker": "leisure_betting",
    "lottery": "leisure_betting",
    "pyrotechnics": "retail_misc",
    "weapons": "retail_misc",
    "military_surplus": "retail_misc",
    "erotic": "retail_misc",
    "cannabis": "retail_misc",
    "e-cigarette": "retail_tobacco",
    "tobacco": "retail_tobacco",
    "hookah": "retail_tobacco",
    "religion": "retail_misc",
    "psychic": "retail_misc",
    "funeral": "retail_misc",
    
    # ==================== ARTESANÍA Y ESPECIALIDADES ====================
    "craft": "craft",
    "fabric": "craft",
    "haberdashery": "craft",
    "wool": "craft",
    "pottery": "craft",
    "knives": "craft",
    
    # ==================== VACANTES Y SIN CATEGORÍA ====================
    "vacant": "other",
    "yes": "other",
    "outpost": "other",

    # ==================== RENTAL ====================
    # "rental": "service_rental",
    "rental_car": "vehicle_car",
    "rental_bicycle": "vehicle_bicycle",
    "rental_motorcycle": "vehicle_motorcycle",
    "rental_vehicle": "vehicle_other",
    "rental_ski": "sport_general",
    "rental_skiing": "sport_general",
    "rental_boat": "sport_general",
}

def extract_poi_category(row):
    """
    Extrae la categoría del POI desde las diferentes columnas de tags.
    Prioriza: shop > amenity > healthcare
    Para shop=rental, intenta inferir el tipo específico desde otros tags.
    """
    # Caso especial: shop=rental
    if row.get('shop') == 'rental':
        # Buscar pistas en otros tags comunes
        rental_type = None
        
        # Verificar tags de tipo de vehículo/servicio
        if pd.notna(row.get('rental')):
            rental_type = row['rental']  # ej: rental=car, rental=bicycle
        elif pd.notna(row.get('service:bicycle:rental')):
            rental_type = 'bicycle'
        elif pd.notna(row.get('service:vehicle:rental')):
            rental_type = 'vehicle'
        elif pd.notna(row.get('sport')):
            rental_type = row['sport']  # ej: sport=skiing
        
        if rental_type:
            return f"shop_rental_{rental_type}"
        else:
            return "shop_rental"
    
    # Verificar shop primero (más específico)
    if pd.notna(row.get('shop')):
        return f"shop_{row['shop']}"
    
    # Luego amenity
    if pd.notna(row.get('amenity')):
        return row['amenity']
    
    # Finalmente healthcare
    if pd.notna(row.get('healthcare')):
        return row['healthcare']
    
    return 'other'

def normalize_category(category, mapping=CATEGORY_MAPPING):
    """
    Normaliza la categoría usando el diccionario de mapeo.
    Si no está en el mapeo, devuelve la categoría original.
    """
    # Si viene con prefijo shop_, extraer la subcategoría
    if category.startswith('shop_'):
        subcategory = category.replace('shop_', '')
        if subcategory in mapping:
            return mapping[subcategory]
        return category  # mantener shop_* si no hay mapeo
    
    # Buscar en el mapeo
    return mapping.get(category, category)

def create_poi_dummies(gdf, mapping=CATEGORY_MAPPING, prefix='poi_'):
    """
    Crea columnas dummy para cada categoría de POI, agrupando según el mapeo.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame con los POIs
    mapping : dict
        Diccionario de mapeo de categorías
    prefix : str
        Prefijo para las columnas dummy
    
    Returns:
    --------
    GeoDataFrame con columnas dummy añadidas
    """
    # Hacer una copia para no modificar el original
    gdf_copy = gdf.copy()
    
    # Extraer categoría principal
    gdf_copy['poi_category_raw'] = gdf_copy.apply(extract_poi_category, axis=1)
    
    # Normalizar/agrupar categorías
    gdf_copy['poi_category'] = gdf_copy['poi_category_raw'].apply(
        lambda x: normalize_category(x, mapping)
    )
    
    # Crear dummies
    dummies = pd.get_dummies(gdf_copy['poi_category'], prefix=prefix.rstrip('_'))
    
    # Unir con el GeoDataFrame original
    result = pd.concat([gdf_copy, dummies], axis=1)
    
    # Estadísticas
    print(f"\n📊 Resumen de categorías:")
    print(f"  - Categorías únicas originales: {gdf_copy['poi_category_raw'].nunique()}")
    print(f"  - Categorías después de agrupar: {gdf_copy['poi_category'].nunique()}")
    print(f"  - Columnas dummy creadas: {len(dummies.columns)}")
    
    print(f"\n🏆 Top 10 categorías más frecuentes:")
    print(gdf_copy['poi_category'].value_counts().head(10))
    
    return result


def filtrar_parcelas_por_frente(parcelas, frentes, min_overlap=0.8):
    # Asegurar proyección métrica
    if parcelas.crs != frentes.crs:
        parcelas = parcelas.to_crs(frentes.crs)

    # Overlay para obtener intersecciones
    inter = gpd.overlay(parcelas, frentes, how="intersection")
    
    # Calcular proporción de área que queda dentro del frente
    inter["ratio_area"] = inter.area / inter["geometry"].area

    # Filtrar parcelas con al menos el % requerido
    return inter[inter["ratio_area"] >= min_overlap]

def agregar_estadisticas_por_frente(inter):
    # Calcular estadísticas por id del frente (ajusta si tu columna se llama distinto)
    stats = (
        inter.groupby("frente_id")["calidad"]
        .agg(
            media_calidad_construccion_frente="mean",
            mediana_calidad_construccion_frente="median"
        )
        .reset_index()
    )
    return stats