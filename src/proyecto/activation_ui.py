"""Visualización de valores reales, publicada al terminar cada capa de inferencia."""
import hashlib
import time

import numpy as np
import pandas as pd
import streamlit as st

from proyecto.network_diagram import network_svg
from proyecto.training import load_run


def input_signature(content):
    return hashlib.sha256(content).hexdigest()


def visualization_options(run):
    enabled = st.checkbox('Visualizar activaciones en tiempo real', key=f'visualize-{run}')
    delay = 0.0
    if enabled:
        delay = st.slider('Pausa didáctica entre capas (segundos)', 0.0, 0.5, 0.15, step=0.05, key=f'delay-{run}')
        st.caption('La vista se actualiza al calcular cada capa para esta entrada. La pausa facilita observar el recorrido; no representa el tiempo de cómputo. El color indica activación, no importancia causal.')
    return enabled, delay


def run_inference(run, signature, predictor, enabled, delay):
    trace = []
    config, _, _ = load_run(run)
    # Eliminar cualquier resultado previo antes de empezar, también si la nueva entrada falla.
    st.session_state.pop(f'inference-{run}', None)
    live = st.empty() if enabled else None
    def on_layer(name, values):
        trace.append(dict(name=name, values=np.asarray(values).copy()))
        with live.container():
            st.write(f'Calculada: {name}')
            st.markdown(network_svg(trace, len(trace)-1, config), unsafe_allow_html=True)
            array = trace[-1]['values']
            if array.ndim == 3:
                maximum = max(float(np.abs(array).max()), 1e-12)
                maps = np.clip(array / maximum, 0, 1)
                count = min(8, array.shape[-1])
                st.image([maps[..., i] for i in range(count)], caption=[f'Canal {i}' for i in range(count)], width=100, clamp=True)
        if delay:
            time.sleep(delay)
    probabilities = predictor(on_layer if enabled else None)
    if live is not None:
        live.empty()
    st.session_state[f'inference-{run}'] = dict(signature=signature, trace=trace, config=config)
    return probabilities


def render_trace(run, signature):
    result = st.session_state.get(f'inference-{run}')
    if not result or result['signature'] != signature or not result['trace']:
        return
    trace = result['trace']
    st.subheader('Activaciones de la última ejecución')
    st.caption('Entrada → extracción de características → clasificación → probabilidades. Mapas y activaciones reales; conexiones esquemáticas sin representar pesos. Azul: positivo; naranja: negativo; claro: cercano a cero. Borde dorado: capa seleccionada. Se muestran muestras de mapas, neuronas y vector aplanado; consulta los valores completos debajo.')
    step = st.select_slider('Inspeccionar capa calculada', options=list(range(len(trace))), value=len(trace)-1,
                            format_func=lambda i: trace[i]['name'], key=f'layer-{run}-{signature}')
    st.markdown(network_svg(trace, step, result.get('config', {})), unsafe_allow_html=True)
    entry = trace[step]
    array = entry['values']
    st.write(f"{entry['name']} · Forma: {array.shape} · Mínimo: {array.min():.5g} · Máximo: {array.max():.5g}")
    if array.ndim == 3:
        count = array.shape[-1]
        channel = st.slider('Canal a inspeccionar', 0, count-1, 0, key=f'channel-{run}-{signature}-{step}') if count > 1 else 0
        maximum = max(float(np.abs(array).max()), 1e-12)
        normalized = np.clip(array/maximum, 0, 1)
        st.caption('Mapas de características: intensidad relativa a la activación máxima de esta capa. No son imágenes reconstruidas ni un mapa de atención.')
        st.image([normalized[..., i] for i in range(min(8, count))], caption=[f'Canal {i}' for i in range(min(8, count))], width=100, clamp=True)
        st.write(f'Valores originales del canal {channel}')
        st.dataframe(pd.DataFrame(array[..., channel]))
    else:
        st.bar_chart(pd.DataFrame({'Activación': array.reshape(-1)}))
        with st.expander('Valores de todas las neuronas de esta capa'):
            st.dataframe(pd.DataFrame({'Índice': np.arange(array.size), 'Activación': array.reshape(-1)}))
    if 'Dropout' in entry['name']:
        st.caption('Dropout está desactivado durante la inferencia; conserva los valores de entrada de esta capa.')
