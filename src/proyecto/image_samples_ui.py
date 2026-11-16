"""Galería paginada de imágenes de prueba compatibles con la CNN guardada."""
import io

from PIL import Image
import streamlit as st

from proyecto.cnn import load_dataset
from proyecto.image_samples import COUNT, save_selection, load_selection, saved_selections


@st.cache_data(max_entries=2, show_spinner=False)
def test_images(dataset):
    _, test = load_dataset(dataset)
    return test


def choose_number(key, number):
    st.session_state[key] = number


def select_dataset_image(dataset):
    st.caption('Las 5 000 imágenes se extraen sin repetición del test oficial del dataset del modelo. No participaron en el ajuste de pesos. La selección se conserva en el volumen de Docker.')
    active_key=f'image-selection-{dataset}'
    if st.button('Extraer 5 000 imágenes al azar', key=f'extract-{dataset}'):
        try:
            with st.spinner('Preparando el dataset y generando la selección…'):
                images, _ = test_images(dataset)
                identifier=save_selection(dataset, len(images))
            st.session_state[active_key]=identifier
        except Exception as exc:
            st.error(f'No se pudo extraer la selección: {exc}')
    identifiers=saved_selections(dataset)
    if not identifiers:
        st.info('Pulsa Extraer 5 000 imágenes al azar para crear tu primera selección. Si falta el dataset, se descargará dentro de Docker.')
        return None, None
    latest=st.session_state.get(active_key)
    identifier=st.selectbox('Selección de imágenes guardada', identifiers,
                           index=identifiers.index(latest) if latest in identifiers else 0,
                           format_func=lambda value: f'Selección {value[:8]}', key=f'selection-picker-{dataset}')
    try:
        selection=load_selection(dataset,identifier)
        with st.spinner('Cargando imágenes de prueba…'):
            images, labels=test_images(dataset)
        if len(images) != selection['total']:
            raise ValueError('El dataset disponible no coincide con la selección guardada.')
    except Exception as exc:
        st.error(f'No se pudo abrir la selección: {exc}')
        return None, None
    st.write(f'Selección de {COUNT:,} imágenes únicas · semilla {selection["seed"]}')
    max_page = COUNT // 20
    page_key = f'gallery-page-{identifier}'
    page = int(st.session_state.get(page_key, 1))
    page = max(1, min(page, max_page))
    st.session_state[page_key] = page
    previous_column, page_column, next_column = st.columns([1, 1, 1])
    with previous_column:
        if st.button('←', key=f'gallery-previous-{identifier}',
                     help='Página anterior', disabled=page <= 1,
                     use_container_width=True):
            st.session_state[page_key] = page - 1
            st.rerun()
    with page_column:
        st.markdown(
            f"<div style='text-align:center;padding-top:0.45rem'>"
            f'<strong>{page}</strong> / {max_page}</div>',
            unsafe_allow_html=True,
        )
    with next_column:
        if st.button('→', key=f'gallery-next-{identifier}',
                     help='Página siguiente', disabled=page >= max_page,
                     use_container_width=True):
            st.session_state[page_key] = page + 1
            st.rerun()
    selected_key=f'gallery-image-{identifier}'
    start=(int(page)-1)*20
    st.caption(f'Imágenes {start+1}–{start+20} de {COUNT}. Pulsa Elegir debajo de una imagen o indica su número.')
    columns=st.columns(5)
    for offset in range(20):
        number=start+offset+1
        index=selection['indices'][number-1]
        with columns[offset%5]:
            st.image(images[index], caption=f'#{number} · test {index}', width=90)
            st.button(f'Elegir #{number}', key=f'choose-{identifier}-{number}', on_click=choose_number, args=(selected_key,number))
    number=st.number_input('Imagen de la selección', 1, COUNT, 1, key=selected_key)
    index=selection['indices'][int(number)-1]
    st.write(f'Imagen elegida: #{number} de la selección · índice {index} del test oficial.')
    output=io.BytesIO()
    Image.fromarray(images[index]).save(output, format='PNG')
    return output.getvalue(), int(labels[index])
