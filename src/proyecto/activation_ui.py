"""Visualización de valores reales, publicada al terminar cada capa de inferencia."""
import hashlib
from html import escape
import time

import numpy as np
import pandas as pd
import streamlit as st


def input_signature(content):
    return hashlib.sha256(content).hexdigest()


def visualization_options(run):
    enabled = st.checkbox('Visualizar activaciones en tiempo real', key=f'visualize-{run}')
    delay = 0.0
    if enabled:
        delay = st.slider('Pausa didáctica entre capas (segundos)', 0.0, 0.5, 0.15, step=0.05, key=f'delay-{run}')
        st.caption('La vista se actualiza al calcular cada capa para esta entrada. La pausa facilita observar el recorrido; no representa el tiempo de cómputo. El color indica activación, no importancia causal.')
    return enabled, delay


def layer_values(array):
    array = np.asarray(array)
    return array.mean(axis=(0, 1)) if array.ndim == 3 else array.reshape(-1)


def network_svg(trace, selected=None):
    width, height = max(640, len(trace) * 155), 410
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Activaciones por capa">',
             f'<rect width="{width}" height="{height}" fill="#111827" rx="12"/>']
    for index, entry in enumerate(trace):
        values = layer_values(entry['values'])
        shown = np.linspace(0, len(values)-1, min(12, len(values)), dtype=int)
        x = 78 + index * 155
        if index:
            parts.append(f'<path d="M{x-118},210 H{x-35}" stroke="#94a3b8" stroke-width="2"/><path d="M{x-42},205 L{x-35},210 L{x-42},215" fill="none" stroke="#94a3b8"/>')
        label = escape(entry['name'])
        active = selected == index
        parts.append(f'<text x="{x}" y="26" fill="{"#facc15" if active else "#e2e8f0"}" font-size="12" text-anchor="middle">{label}</text>')
        unit = 'canales (media)' if entry['values'].ndim == 3 else 'valores'
        parts.append(f'<text x="{x}" y="46" fill="#94a3b8" font-size="11" text-anchor="middle">{len(values)} {unit}</text>')
        scale = max(float(np.max(np.abs(values))), 1e-12)
        for row, neuron in enumerate(shown):
            value = float(values[neuron])
            amount = min(abs(value)/scale, 1.0)
            color = (np.array([30, 41, 59])*(1-amount) + np.array([34, 211, 238] if value >= 0 else [251, 146, 60])*amount).astype(int)
            color = '#%02x%02x%02x' % tuple(color)
            y = 75 + row * 24
            parts.append(f'<circle cx="{x}" cy="{y}" r="9" fill="{color}" stroke="{"#facc15" if active else "#64748b"}"><title>Índice {neuron}: {value:.6g}</title></circle>')
            parts.append(f'<text x="{x+14}" y="{y+4}" fill="#cbd5e1" font-size="10">{neuron}: {value:.3g}</text>')
        parts.append(f'<text x="{x}" y="390" fill="#94a3b8" font-size="10" text-anchor="middle">{len(shown)} de {len(values)} visibles</text>')
    parts.append('</svg>')
    return '<div style="overflow-x:auto">' + ''.join(parts) + '</div>'


def run_inference(run, signature, predictor, enabled, delay):
    trace = []
    # Eliminar cualquier resultado previo antes de empezar, también si la nueva entrada falla.
    st.session_state.pop(f'inference-{run}', None)
    live = st.empty() if enabled else None
    def on_layer(name, values):
        trace.append(dict(name=name, values=np.asarray(values).copy()))
        with live.container():
            st.write(f'Calculada: {name}')
            st.markdown(network_svg(trace, len(trace)-1), unsafe_allow_html=True)
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
    st.session_state[f'inference-{run}'] = dict(signature=signature, trace=trace)
    return probabilities


def render_trace(run, signature):
    result = st.session_state.get(f'inference-{run}')
    if not result or result['signature'] != signature or not result['trace']:
        return
    trace = result['trace']
    st.subheader('Activaciones de la última ejecución')
    st.caption('Cian: valor positivo; naranja: negativo; oscuro: cercano a cero. Escala relativa por capa. Se muestran hasta 12 neuronas o medias de canales por columna; las flechas representan el flujo entre capas, no pesos ni conexiones individuales.')
    step = st.select_slider('Inspeccionar capa calculada', options=list(range(len(trace))), value=len(trace)-1,
                            format_func=lambda i: trace[i]['name'], key=f'layer-{run}-{signature}')
    st.markdown(network_svg(trace, step), unsafe_allow_html=True)
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
