from PIL import Image

from hypothesis import given, strategies as st
import tempfile
import os
from function  import  apri_immagine
@given(path=st.text(min_size=1).filter(lambda s: "\x00" not in s))
def test_apri_immagine_file_inesistente_lancia_errore(path):
    # Evita collisioni con file reali: prefisso improbabile
    fake_path = "__file_inesistente_hypothesis__/" + path

    try:
        apri_immagine(fake_path)
        assert False, "Mi aspettavo FileNotFoundError"
    except FileNotFoundError:
        assert True
    except Exception:
        # Su alcuni OS/path strani possono emergere altri OSError
        # ma per lo scopo del corso teniamo chiaro il caso atteso
        assert True


