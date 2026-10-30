"""Diagrama didáctico de activaciones con entrada, extracción y clasificación."""
import base64
from html import escape
import io

import numpy as np
from PIL import Image

BLUE = '#07649c'
DARK = '#064d86'
PALE = '#e7f2f7'


def bitmap(values, scale=None):
    """PNG de los valores reales; escala relativa dentro del mapa mostrado."""
    values = np.asarray(values, dtype=float)
    maximum = max(float(np.abs(values).max()) if scale is None else scale, 1e-12)
    pixels = np.uint8(np.clip(values / maximum, 0, 1) * 255)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


def text(x, y, value, size=15, color=DARK, anchor='middle', weight='normal'):
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{escape(str(value))}</text>'


def image(values, x, y, width, height, active=False, title='', scale=None):
    stroke = '#d99b13' if active else BLUE
    return (f'<g><title>{escape(title)}</title><rect x="{x-2}" y="{y-2}" width="{width+4}" height="{height+4}" fill="white" stroke="{stroke}" stroke-width="{4 if active else 2}"/>'
            f'<image href="{bitmap(values, scale)}" x="{x}" y="{y}" width="{width}" height="{height}" preserveAspectRatio="none" style="image-rendering:pixelated"/></g>')


def line(x1, y1, x2, y2, opacity=1, arrow=False):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{BLUE}" stroke-width="1" opacity="{opacity}" {"marker-end=\"url(#arrow)\"" if arrow else ""}/>'


def node_color(value, scale):
    amount = min(abs(float(value)) / max(scale, 1e-12), 1)
    rgb = np.array([225, 237, 244]) * (1-amount) + np.array([4, 90, 153] if value >= 0 else [220, 115, 38]) * amount
    return '#%02x%02x%02x' % tuple(rgb.astype(int))


def neuron_column(values, x, selected=False, limit=8):
    values = np.asarray(values).reshape(-1)
    indices = np.linspace(0, len(values)-1, min(limit, len(values)), dtype=int)
    ys = np.linspace(190, 420, len(indices))
    scale = float(np.abs(values).max())
    result = []
    for index, y in zip(indices, ys):
        result.append(f'<circle cx="{x}" cy="{y}" r="10" fill="{node_color(values[index], scale)}" stroke="{"#d99b13" if selected else DARK}" stroke-width="{3 if selected else 1}"><title>Neurona {index}: {values[index]:.6g}</title></circle>')
    return ''.join(result), ys


def start(width, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="580" viewBox="0 0 {width} 580" role="img" aria-label="{title}" style="font-family:Arial,sans-serif;max-width:100%;min-width:960px;height:auto">',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#07649c"/></marker></defs>',
            f'<rect width="{width}" height="580" rx="16" fill="#fafcfd"/>', text(30, 49, title, 29, anchor='start', weight='bold')]


def finish(parts):
    parts.append('</svg>')
    return '<div style="overflow-x:auto;width:100%">' + ''.join(parts) + '</div>'


def cnn_svg(trace, selected, config):
    # Las capas espaciales se muestran como pilas; Dropout no agrega otra columna.
    spatial = [(i, entry) for i, entry in enumerate(trace[1:], 1) if entry['values'].ndim == 3]
    flatten = next(((i, entry) for i, entry in enumerate(trace) if 'Flatten' in entry['name']), None)
    output_index = 2 * len(config.get('filters', [32, 64])) + 4
    dense = [(i, entry) for i, entry in enumerate(trace) if 'Dense' in entry['name'] and i != output_index]
    width = max(1280, 740 + 165 * len(spatial) + 90 * max(0, len(dense)-1))
    output_x, flat_x = width-260, 260+165*len(spatial)
    parts = start(width, 'RED NEURONAL CONVOLUCIONAL (CNN)')
    parts += [f'<rect x="18" y="95" width="205" height="385" rx="34" fill="{PALE}"/>',
              f'<rect x="{output_x}" y="95" width="242" height="385" rx="34" fill="{PALE}"/>',
              text(120, 138, 'ENTRADA', 25), text(output_x+121, 138, 'SALIDA', 25)]
    input_array = trace[0]['values']
    parts.append(image(input_array[..., 0], 43, 200, 154, 154, selected == 0, 'Imagen de entrada normalizada', scale=1.0))
    parts += [text(120, 390, '28 × 28 · escala de grises', 12), text(120, 410, 'Píxeles normalizados [0, 1]', 12)]
    parts.append(line(225, 297, 248, 297, arrow=True))
    last_right = 223
    for stage, (index, entry) in enumerate(spatial):
        array = entry['values']
        pooling = 'Pooling' in entry['name']
        size = max(54, min(112, int(array.shape[0]*4.3)))
        x = 261+165*stage
        y = 300-size/2
        channels = np.linspace(0, array.shape[-1]-1, min(4, array.shape[-1]), dtype=int)
        # Una escala común por capa permite comparar las intensidades de sus filtros.
        maximum = max(float(np.abs(array).max()), 1e-12)
        for depth, channel in enumerate(channels):
            offset = depth * 9
            # bitmap normaliza por mapa; añadir el máximo común conserva la escala entre canales.
            plane = np.asarray(array[..., channel]) / maximum
            # Codificar directamente la escala de la capa (sin renormalizar el canal).
            stream = io.BytesIO()
            Image.fromarray(np.uint8(np.clip(plane, 0, 1)*255)).save(stream, format='PNG')
            uri = 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')
            px, py = x+offset, y+offset
            stroke = '#d99b13' if index == selected else BLUE
            parts += [f'<g><title>Canal {channel} · {array.shape[0]}×{array.shape[1]}</title>',
                      f'<rect x="{px-2}" y="{py-2}" width="{size+4}" height="{size+4}" fill="white" stroke="{stroke}" stroke-width="{4 if index == selected else 2}"/>',
                      f'<image href="{uri}" x="{px}" y="{py}" width="{size}" height="{size}" preserveAspectRatio="none" style="image-rendering:pixelated"/></g>']
        center = x+size/2+13
        parts += [text(center, 166, 'Agrupación' if pooling else 'Convolución', 14),
                  text(center, 439, f'{array.shape[0]}×{array.shape[1]} · {array.shape[-1]} canales', 12),
                  text(center, 459, f'{len(channels)} mapas visibles', 11, '#526678')]
        if stage:
            parts.append(line(last_right+6, 306, x-7, 306, opacity=0.6, arrow=True))
        last_right = x+size+27
    if spatial:
        parts.append(line(last_right+6, 306, flat_x-6, 306, opacity=0.6, arrow=True))
    if flatten:
        index, entry = flatten
        vector = entry['values'].reshape(-1)
        samples = np.linspace(0, len(vector)-1, min(220, len(vector)), dtype=int)
        parts.append(image(vector[samples, None], flat_x, 190, 29, 240, selected == index, 'Muestra ordenada del vector real aplanado'))
        parts += [text(flat_x+15, 166, 'Aplanamiento', 13), text(flat_x+15, 454, f'{len(vector)} valores', 11)]
    else:
        parts += [f'<rect x="{flat_x}" y="190" width="29" height="240" fill="{PALE}" stroke="#afc5d3" stroke-dasharray="4 4"/>', text(flat_x+15, 454, 'Aplanamiento', 11)]
    # Mostrar conexiones esquemáticas entre muestras de neuronas, sin inventar pesos.
    previous_x, previous_ys = flat_x+29, np.linspace(200, 420, 8)
    dense_x = flat_x+102
    for column, (index, entry) in enumerate(dense):
        x = dense_x+column*85
        nodes, ys = neuron_column(entry['values'], x, selected == index or (selected is not None and selected == index+1 and selected < len(trace) and 'Dropout' in trace[selected]['name']))
        for y1 in previous_ys:
            for y2 in ys:
                parts.append(line(previous_x+9, y1, x-11, y2, opacity=0.23))
        parts.append(nodes)
        parts += [text(x, 166, 'Densa · ReLU', 13), text(x, 454, f'{entry["values"].size} neuronas', 11), text(x, 474, 'Dropout inactivo al predecir', 10, '#526678')]
        previous_x, previous_ys = x, ys
    if not dense:
        parts.append(text(dense_x, 305, 'Capas densas', 13, '#71899a'))
    output = trace[output_index]['values'].reshape(-1) if len(trace) > output_index else None
    if output is not None:
        out_x = output_x-38
        nodes, ys = neuron_column(output, out_x, selected == output_index, limit=10)
        for y1 in previous_ys:
            for y2 in ys:
                parts.append(line(previous_x+11, y1, out_x-11, y2, opacity=0.18))
        parts.append(nodes)
        classes = config.get('classes', [str(i) for i in range(len(output))])
        winner = int(np.argmax(output))
        for i, (name, probability) in enumerate(zip(classes, output)):
            y = 177+i*27
            parts.append(text(output_x+14, y, name, 12, anchor='start', weight='bold' if i == winner else 'normal'))
            parts.append(f'<rect x="{output_x+114}" y="{y-10}" width="80" height="11" rx="3" fill="#cddfe9"/>')
            parts.append(f'<rect x="{output_x+114}" y="{y-10}" width="{80*float(probability):.3f}" height="11" rx="3" fill="{DARK if i == winner else BLUE}"/>')
            parts.append(text(output_x+232, y, f'{float(probability):.1%}', 11, anchor='end'))
        parts.append(text(output_x+121, 463, f'Predicción: {classes[winner]}', 13, weight='bold'))
    else:
        parts += [text(output_x+121, 280, 'Probabilidades', 18), text(output_x+121, 306, 'Esperando la capa de salida', 13, '#526678')]
    for left, right, label in [(28, flat_x+30, 'Extracción de características'), (flat_x+46, output_x-20, 'Clasificación'), (output_x+5, width-20, 'Probabilidades')]:
        parts += [line(left, 517, right, 517, arrow=True), text((left+right)/2, 544, label, 14, '#526678')]
    parts.append(text(28, 568, 'Mapas y valores reales · conexiones esquemáticas · borde dorado: capa seleccionada', 11, '#526678', anchor='start'))
    return finish(parts)


def dense_svg(trace, selected, config):
    # Conservar cada transformación para que el diagrama coincida con el inspector.
    columns = list(enumerate(trace))
    width = max(1000, 400+150*len(columns))
    parts = start(width, 'RED NEURONAL DENSA (MLP)')
    parts += [f'<rect x="18" y="95" width="175" height="385" rx="30" fill="{PALE}"/>',
              f'<rect x="{width-225}" y="95" width="207" height="385" rx="30" fill="{PALE}"/>']
    positions = np.linspace(105, width-300, len(columns))
    previous = None
    for (index, entry), x in zip(columns, positions):
        values = entry['values'].reshape(-1)
        nodes, ys = neuron_column(values, x, index == selected, limit=10)
        if previous:
            px, py = previous
            for y1 in py:
                for y2 in ys:
                    parts.append(line(px+11, y1, x-11, y2, opacity=0.17))
        parts += [nodes, text(x, 146, 'ENTRADA' if index == 0 else ('SALIDA' if 'probabilidades' in entry['name'] else entry['name']), 15),
                  text(x, 456, f'{len(values)} valores · {len(ys)} visibles', 11)]
        previous = x, ys
    if 'probabilidades' in trace[-1]['name']:
        values = trace[-1]['values'].reshape(-1)
        classes = config.get('classes', [str(i) for i in range(len(values))])
        winner = int(np.argmax(values))
        top = np.argsort(-values)[:10]
        for row, i in enumerate(top):
            y = 180+row*26
            parts += [text(width-210, y, classes[i], 12, anchor='start'), text(width-30, y, f'{float(values[i]):.1%}', 12, anchor='end')]
        parts += [text(width-120, 140, 'PROBABILIDADES', 16), text(width-120, 458, f'Predicción: {classes[winner]}', 12, weight='bold')]
    else:
        parts.append(text(width-120, 290, 'Calculando…', 16))
    parts += [line(30, 517, width-240, 517, arrow=True), text((width-240)/2, 544, 'Propagación de activaciones', 15, '#526678'),
              text(28, 569, 'Azul: positivo · naranja: negativo · escala relativa por capa · conexiones esquemáticas', 11, '#526678', anchor='start')]
    return finish(parts)


def network_svg(trace, selected=None, config=None):
    config = config or {}
    if trace[0]['values'].ndim == 3:
        return cnn_svg(trace, selected, config)
    return dense_svg(trace, selected, config)
