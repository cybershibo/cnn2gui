import streamlit as st
from proyecto.compute import probe

@st.cache_data(ttl=60)
def devices(framework):
    return probe(framework)

def device_selector(framework, key):
    available, reason = devices(framework)
    names = {'cpu': 'CPU'}
    names.update({code: f'GPU NVIDIA · {name} ({code})' for code, name in available})
    choice=st.selectbox('Dispositivo de cómputo', list(names), format_func=names.get, key=key)
    if not available:
        st.caption('CUDA no disponible: '+(reason or 'No se detectó una GPU NVIDIA utilizable en el contenedor.'))
    return choice
