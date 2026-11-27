
from pathlib import Path

import numpy as np
import pandas as pd


def load_data_csv(file_path: str | Path, sep: str = ',', decimal: str = '.', encoding: str = 'utf-8') -> pd.DataFrame:
    try:
        return pd.read_csv(file_path, sep=sep, decimal=decimal, encoding=encoding)
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        return pd.DataFrame()


def save_data_csv(df: pd.DataFrame,file_path: str | Path,sep: str = ',',decimal: str = '.',encoding: str = 'utf-8',index: bool = False) -> bool:
    try:
        df.to_csv(file_path, sep=sep, decimal=decimal, encoding=encoding,index=index)
        return True

    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")
        return False


    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")
        return False


def load_json(file_path: str | Path) -> dict | list:
    """Carga un archivo JSON y devuelve su contenido."""
    import json

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)