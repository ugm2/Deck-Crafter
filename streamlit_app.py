import streamlit as st
import requests
import json
from typing import Dict, Any
import html

# Helper function to clean and unescape text
def clean_and_unescape_text(text_val: Any) -> str:
    if text_val is None:
        return "No especificado"
    # Ensure it's a string, then remove markdown bold/italics and unescape HTML entities
    return html.unescape(str(text_val).replace('**', '').replace('*', ''))

# Helper function to build a list item for card details
def _build_card_detail_li(label: str, value: Any) -> str:
    return f'<li><b>{label}:</b> {clean_and_unescape_text(value)}</li>'

# Helper function to generate full card HTML
def _generate_card_html(card: Dict[str, Any], rarity_class: str) -> str:
    name = clean_and_unescape_text(card.get("name", "Sin nombre"))
    rarity = clean_and_unescape_text(card.get("rarity", "Común"))
    description = clean_and_unescape_text(card.get("description", "Sin descripción"))

    cost_li = ""
    if card.get('cost') is not None:
        cost_li = _build_card_detail_li("Costo", card.get("cost"))

    list_items_html = ""
    list_items_html += _build_card_detail_li("Cantidad", card.get("quantity")) + "\n"
    list_items_html += _build_card_detail_li("Tipo", card.get("type")) + "\n"
    if cost_li:
        list_items_html += cost_li + "\n"
    list_items_html += _build_card_detail_li("Texto de Sabor", card.get("flavor_text")) + "\n"
    list_items_html += _build_card_detail_li("Interacciones", card.get("interactions")) + "\n"
    list_items_html += _build_card_detail_li("Color", card.get("color")) + "\n"
    list_items_html += _build_card_detail_li("Descripción de Imagen", card.get("image_description")) + "\n"

    # Add image if available
    image_html = ""
    if card.get("image_data"):
        image_html = f'<div class="card-image"><img src="data:image/png;base64,{card["image_data"]}" alt="{name}" style="width: 100%; max-height: 200px; object-fit: contain;"></div>\n'

    # Construct the full card HTML string using minimal f-string for the main structure
    card_html = (
        '<div class="card">\n'
        f'{image_html}'  # Add image at the top
        '    <div class="card-title">\n'
        f'        {name} \n'
        f'        <span class="{rarity_class}">({rarity})</span>\n'
        '    </div>\n'
        '    <div class="card-content">\n'
        f'        <p>{description}</p>\n'
        '        <ul style="list-style-type: none; padding-left: 0;">\n'
        f'{list_items_html}'  # Embed pre-built list HTML
        '        </ul>\n'
        '    </div>\n'
        '</div>\n'
    )
    return card_html

if "_pending_preferences" in st.session_state:
    for k, v in st.session_state["_pending_preferences"].items():
        st.session_state[k] = v
    del st.session_state["_pending_preferences"]
    st.session_state["_just_updated_preferences"] = True
    st.rerun()

st.set_page_config(
    page_title="Deck Crafter",
    page_icon="🎮",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .game-title {
        font-size: 2.8rem;
        font-weight: bold;
        color: #306998;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .section-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #4B8BBE;
        margin: 2rem 0 1.2rem 0;
        border-bottom: 2px solid #4B8BBE;
        padding-bottom: 0.5rem;
    }
    .content-container {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #F0F0F0;
    }
    .info-panel {
        background-color: #F0F8FF; /* A light blue background */
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #CCE0FF;
    }
    .card {
        background-color: #F8F8F8; /* Light gray background */
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); /* Deeper shadow */
        border: 1px solid #E0E0E0; /* Subtle border */
        width: 100%; 
        display: block; 
        transition: transform 0.2s ease-in-out; /* Smooth hover effect */
    }
    .card:hover {
        transform: translateY(-5px); /* Lift effect on hover */
    }
    .card-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2B3A42;
        margin-bottom: 0.6rem;
        text-align: center;
    }
    .card-content {
        color: #2B3A42;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    .rarity-common { color: #555555; font-weight: bold; }
    .rarity-uncommon { color: #2C7DA0; font-weight: bold; }
    .rarity-rare { color: #DAA520; font-weight: bold; }
    .rarity-raro { color: #DAA520; font-weight: bold; }
    .rarity-legendario { color: #8B0000; font-weight: bold; }
    .rarity-mythic { color: #8A2BE2; font-weight: bold; }
    .concept-main {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #F0F0F0;
    }
    .card-type-item {
        background-color: #F8F8F8;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #E0E0E0;
    }
    .card-image {
        margin-bottom: 1rem;
        text-align: center;
        background-color: #FFFFFF;
        padding: 0.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

if 'game_id' not in st.session_state:
    st.session_state.game_id = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'start'

st.markdown('<div class="game-title">🎮 Deck Crafter</div>', unsafe_allow_html=True)
st.markdown("Crea tu propio juego de cartas con IA")

with st.expander("Cargar un Juego Existente"):
    game_id_input = st.text_input(
        "Introduce el ID de un juego para cargarlo",
        placeholder="Pega el ID del juego aquí"
    )
    load_game = st.button("Cargar Juego")

game_description = st.text_area(
    "Describe tu juego de cartas",
    placeholder="Describe el tipo de juego que te gustaría crear...",
    help="Describe tu idea de juego. Por ejemplo: 'Un juego de cartas de fantasía donde los jugadores son magos que compiten por dominar diferentes escuelas de magia'"
)

with st.expander("Preferencias del Juego (Opcional - Si no las especificas, se generarán automáticamente)"):
    st.text_input(
        "Idioma",
        placeholder="Ejemplo: Español, English",
        key="language"
    )
    st.text_input(
        "Tema",
        placeholder="Ejemplo: Fantasía, Ciencia ficción, Medieval...",
        key="theme"
    )
    st.text_input(
        "Estilo de Juego",
        placeholder="Ejemplo: estrategia, suerte, mixto",
        key="game_style"
    )
    st.text_input(
        "Número de Jugadores",
        placeholder="Ejemplo: 2-4",
        key="number_of_players"
    )
    st.text_input(
        "Audiencia Objetivo",
        placeholder="Ejemplo: niños, adolescentes, adultos, todas las edades",
        key="target_audience"
    )
    st.text_input(
        "Complejidad de las Reglas",
        placeholder="Ejemplo: simple, media, compleja",
        key="rule_complexity"
    )

start_game = st.button("Iniciar Juego")

def call_api(endpoint: str, method: str = "POST", data: Dict[str, Any] = None) -> Dict[str, Any]:
    try:
        url = f"http://localhost:8000/api/v1/games/{endpoint}"
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error al llamar a la API: {str(e)}")
        return None

if start_game:
    if not game_description:
        st.error("Por favor, proporciona una descripción del juego que quieres crear.")
    else:
        data = {
            "game_description": game_description,
            "language": st.session_state.get("language") or None,
            "theme": st.session_state.get("theme") or None,
            "game_style": st.session_state.get("game_style") or None,
            "number_of_players": st.session_state.get("number_of_players") or None,
            "target_audience": st.session_state.get("target_audience") or None,
            "rule_complexity": st.session_state.get("rule_complexity") or None
        }
        
        with st.spinner("Iniciando juego y generando concepto y reglas..."):
            result = call_api("start", data=data)
            if result and "game_id" in result:
                st.session_state.game_id = result["game_id"]
                game_state = call_api(f"{st.session_state.game_id}", method="GET")
                if game_state and "preferences" in game_state:
                    st.session_state["_pending_preferences"] = game_state["preferences"]
                
                concept_rules_result = call_api(f"{st.session_state.game_id}/concept-and-rules")
                if concept_rules_result and concept_rules_result.get("status") == "rules_generated":
                    st.session_state.current_step = "cards"
                    st.success("¡Juego iniciado! Ahora puedes generar las cartas.")
                    st.rerun()
            else:
                st.session_state.game_id = None
                st.session_state.current_step = 'start'

if load_game:
    if not game_id_input:
        st.error("No has proporcionado ningún ID válido")
    st.session_state.game_id = game_id_input

if st.session_state.game_id:
    game_state = call_api(f"{st.session_state.game_id}", method="GET")
    if game_state:
        # Game Concept Section
        if game_state.get("concept"):
            concept = game_state["concept"]
            st.markdown('<div class="section-title">🎲 Concepto del Juego</div>', unsafe_allow_html=True)
            
            concept_cols = st.columns([2, 1])
            
            with concept_cols[0]:
                st.markdown(f"""
                    <div class="card">
                        <div class="card-title">{clean_and_unescape_text(concept.get("title", "Sin título"))}</div>
                        <p>{clean_and_unescape_text(concept.get("description", ""))}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with concept_cols[1]:
                st.markdown('<div class="info-panel">', unsafe_allow_html=True) # Use new info-panel class
                st.markdown("<b>Detalles del Juego:</b>", unsafe_allow_html=True)
                st.markdown(f"""🗣️ <b>Idioma:</b> {clean_and_unescape_text(concept.get("language", "No especificado"))}""", unsafe_allow_html=True)
                st.markdown(f"""👥 <b>Jugadores:</b> {clean_and_unescape_text(concept.get("number_of_players", "No especificado"))}""", unsafe_allow_html=True)
                st.markdown(f"""🎯 <b>Audiencia:</b> {clean_and_unescape_text(concept.get("target_audience", "No especificado"))}""", unsafe_allow_html=True)
                st.markdown(f"""🎨 <b>Tema:</b> {clean_and_unescape_text(concept.get("theme", "No especificado"))}""", unsafe_allow_html=True)
                st.markdown(f"""🎮 <b>Estilo:</b> {clean_and_unescape_text(concept.get("game_style", "No especificado"))}""", unsafe_allow_html=True)
                st.markdown(f"""⏳ <b>Duración Estimada:</b> {clean_and_unescape_text(concept.get("game_duration", "No especificado"))}""", unsafe_allow_html=True)
                st.markdown(f"""📚 <b>Cartas Únicas Requeridas:</b> {clean_and_unescape_text(concept.get("number_of_unique_cards", "No especificado"))}""", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if concept.get("card_types"):
                st.markdown('<div class="section-title">🏷️ Tipos de Cartas</div>', unsafe_allow_html=True)
                for card_type in concept["card_types"]:
                    st.markdown(f"""
                        <div class="card-type-item">
                            <div class="card-type-name">{clean_and_unescape_text(card_type.get("name", "Sin nombre"))}</div>
                            <p>{clean_and_unescape_text(card_type.get("description", ""))}</p>
                            <p><b>Cantidad:</b> {clean_and_unescape_text(card_type.get("quantity"))} | <b>Cartas Únicas:</b> {clean_and_unescape_text(card_type.get("unique_cards"))}</p>
                        </div>
                    """, unsafe_allow_html=True)

        # Rules Section
        if game_state.get("rules"):
            rules = game_state["rules"]
            st.markdown('<div class="section-title">📖 Reglas del Juego</div>', unsafe_allow_html=True)
            
            tabs = st.tabs(["Configuración", "Jugabilidad", "Glosario", "Ejemplos de Juego", "Otras Reglas"])
            
            # Pestaña 0: Configuración (Setup)
            with tabs[0]:
                st.markdown('<div class="content-container">', unsafe_allow_html=True)
                st.subheader("Preparación de la Partida")
                st.markdown(f"<p><b>Mazo:</b> {clean_and_unescape_text(rules.get('deck_preparation', 'No especificado'))}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Mano Inicial:</b> {clean_and_unescape_text(rules.get('initial_hands', 'No especificado'))}</p>", unsafe_allow_html=True)
                st.subheader("Condiciones de Victoria")
                st.markdown(f"<p>{clean_and_unescape_text(rules.get('win_conditions', 'No especificado'))}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Pestaña 1: Jugabilidad (Gameplay)
            with tabs[1]:
                st.markdown('<div class="content-container">', unsafe_allow_html=True)
                st.subheader("Estructura del Turno")
                # --- NUEVA LÓGICA PARA MOSTRAR LAS FASES DEL TURNO ---
                if rules.get("turn_structure"):
                    for phase in rules["turn_structure"]:
                        st.markdown(f"<h6>{clean_and_unescape_text(phase.get('phase_name', 'Fase sin nombre'))}</h6>", unsafe_allow_html=True)
                        st.markdown(f"<p>{clean_and_unescape_text(phase.get('phase_description', ''))}</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p>No se especificó una estructura de turno detallada.</p>", unsafe_allow_html=True)
                
                st.subheader("Mecánicas Adicionales")
                st.markdown(f"<p><b>Fase de Reacción:</b> {clean_and_unescape_text(rules.get('reaction_phase', 'No especificada'))}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Fin de Ronda:</b> {clean_and_unescape_text(rules.get('end_of_round', 'No especificado'))}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Mecánicas de Recursos:</b> {clean_and_unescape_text(rules.get('resource_mechanics', 'No especificadas'))}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # --- NUEVA PESTAÑA PARA EL GLOSARIO ---
            with tabs[2]:
                st.markdown('<div class="content-container">', unsafe_allow_html=True)
                st.subheader("Glosario de Términos")
                if rules.get("glossary"):
                    for term, definition in rules["glossary"].items():
                        st.markdown(f"<b>{clean_and_unescape_text(term)}:</b> {clean_and_unescape_text(definition)}", unsafe_allow_html=True)
                else:
                    st.markdown("<p>No hay un glosario disponible.</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # --- NUEVA PESTAÑA PARA EJEMPLOS ---
            with tabs[3]:
                st.markdown('<div class="content-container">', unsafe_allow_html=True)
                st.subheader("Ejemplos de Juego")
                if rules.get("examples_of_play"):
                    for i, example in enumerate(rules["examples_of_play"]):
                        st.markdown(f"<b>Ejemplo {i+1}:</b>", unsafe_allow_html=True)
                        st.markdown(f"<p><i>{clean_and_unescape_text(example)}</i></p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p>No se proporcionaron ejemplos.</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # Pestaña 4: Otras Reglas
            with tabs[4]:
                st.markdown('<div class="content-container">', unsafe_allow_html=True)
                st.subheader("Detalles y Reglas Adicionales")
                st.markdown(f"<p><b>Límite de Turnos:</b> {clean_and_unescape_text(rules.get('turn_limit', 'No especificado'))}</p>", unsafe_allow_html=True)
                st.markdown(f"<p><b>Sistema de Puntuación:</b> {clean_and_unescape_text(rules.get('scoring_system', 'No especificado'))}</p>", unsafe_allow_html=True)
                if rules.get("additional_rules"):
                    st.markdown("<b>Reglas Misceláneas:</b>", unsafe_allow_html=True)
                    for rule in rules["additional_rules"]:
                        st.markdown(f"<p>* {clean_and_unescape_text(rule)}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Cards Section
        if game_state.get("cards"):
            st.markdown('<div class="section-title">🃏 Cartas del Juego</div>', unsafe_allow_html=True)
            
            # Group cards by type
            cards_by_type = {}
            for card in game_state["cards"]:
                card_type = card.get("type", "Sin tipo")
                if card_type not in cards_by_type:
                    cards_by_type[card_type] = []
                cards_by_type[card_type].append(card)
            
            # Create tabs for each card type
            card_type_tabs = st.tabs(list(cards_by_type.keys()))
            
            for tab, card_type in zip(card_type_tabs, cards_by_type.keys()):
                with tab:
                    cards = cards_by_type[card_type]
                    # Display cards in a 5-column layout
                    cols = st.columns(5) # Changed to 5 columns
                    col_idx = 0
                    for card in cards:
                        with cols[col_idx]:
                            rarity = card.get('rarity', 'Común')
                            rarity = rarity if rarity else "Común"
                            rarity_class = f"rarity-{rarity.lower().replace(' ', '')}"
                            
                            # Generate card HTML using the helper function
                            card_display_html = _generate_card_html(card, rarity_class)
                            st.markdown(card_display_html, unsafe_allow_html=True)
                        col_idx = (col_idx + 1) % 5

        # Evaluation Section
        if game_state.get("evaluation"):
            st.markdown('<div class="section-title">🏆 Evaluación del Juego</div>', unsafe_allow_html=True)
            evaluation = game_state["evaluation"]

            st.subheader("Resumen General (Puntuación: {overall_score}/10)".format(overall_score=evaluation.get('overall_score', 'N/A')))
            st.info(evaluation.get('summary', 'No hay resumen disponible.'))

            with st.expander("Ver Análisis Detallado"):
                # Mapeo de las claves del modelo a títulos amigables para la interfaz
                metric_map = {
                    "balance": "Balance",
                    "coherence": "Coherencia",
                    "clarity": "Claridad",
                    "originality": "Originalidad",
                    "playability": "Jugabilidad",
                    "fidelity": "Fidelidad a la Petición"
                }

                for key, title in metric_map.items():
                    # Asegurarse de que la clave exista antes de intentar acceder
                    if key in evaluation:
                        metric_data = evaluation[key]
                        st.subheader(f"{title} (Puntuación: {metric_data.get('score', 'N/A')}/10)")
                        st.markdown(f"**Análisis:** {metric_data.get('analysis', 'No hay análisis disponible.')}")
                        if metric_data.get("suggestions"): # Usar .get() para evitar KeyError si 'suggestions' no existe
                            st.markdown("**Sugerencias de Mejora:**")
                            for suggestion in metric_data["suggestions"]:
                                st.markdown(f"- {suggestion}")
                        st.divider()

        # Add evaluation button if game is ready for evaluation
        if game_state.get("status") in ["cards_generated", "images_generated"] and not game_state.get("evaluation"):
            if st.button("🤖 Evaluar Juego"):
                with st.spinner("Un experto en diseño de juegos está analizando tu creación..."):
                    result = call_api(f"{st.session_state.game_id}/evaluate")
                    if result and result.get("status") == "evaluated":
                        st.success("¡Evaluación completada! Los resultados se muestran arriba.")
                        st.rerun()

    if st.session_state.current_step == "cards":
        if st.button("Generar Cartas"):
            with st.spinner("Generando cartas..."):
                result = call_api(f"{st.session_state.game_id}/cards")
                if result and result.get("status") == "cards_generated":
                    st.session_state.current_step = "images"
                    st.success("¡Cartas generadas! Ahora puedes generar las imágenes.")
                    st.rerun()

if st.session_state.current_step == "images":
    if st.button("Generar Imágenes"):
        with st.spinner("Generando imágenes..."):
            result = call_api(f"{st.session_state.game_id}/images")
            if result and result.get("status") == "images_generated":
                st.session_state.current_step = "complete"
                st.success("¡Juego completado! Puedes ver el resultado completo arriba.")
                st.rerun()
