import streamlit as st
import requests
import json
from typing import Dict, Any
import html
import time

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
if 'premium_mode' not in st.session_state:
    st.session_state.premium_mode = False
if 'premium_provider' not in st.session_state:
    st.session_state.premium_provider = "gemini"
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = "gemini-3.1-pro-preview"

# Available Gemini models
GEMINI_MODELS = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview (Recomendado)",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "gemini-2.5-pro-preview-06-05": "Gemini 2.5 Pro",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
}

# Premium mode toggle in sidebar
with st.sidebar:
    st.subheader("Modo de Generación")
    premium_mode = st.toggle(
        "Modo Premium",
        value=st.session_state.premium_mode,
        help="Usa modelos premium para mayor calidad"
    )
    st.session_state.premium_mode = premium_mode

    if premium_mode:
        premium_provider = st.selectbox(
            "Proveedor Premium",
            options=["gemini", "groq"],
            format_func=lambda x: "Gemini" if x == "gemini" else "Groq GPT-OSS-20B",
            index=0 if st.session_state.premium_provider == "gemini" else 1,
        )
        st.session_state.premium_provider = premium_provider

        # Show model selector for Gemini
        if premium_provider == "gemini":
            model_options = list(GEMINI_MODELS.keys())
            current_idx = model_options.index(st.session_state.gemini_model) if st.session_state.gemini_model in model_options else 0
            gemini_model = st.selectbox(
                "Modelo Gemini",
                options=model_options,
                format_func=lambda x: GEMINI_MODELS.get(x, x),
                index=current_idx,
                help="Gemini 2.5 Pro ofrece mejor calidad pero es más lento"
            )
            st.session_state.gemini_model = gemini_model

        st.warning(f"⚠️ Modo Premium: {premium_provider.upper()}")
    else:
        st.info("Modo estándar (gratuito)")

    st.markdown("---")

    # Refinement controls
    st.subheader("Refinamiento Iterativo")
    st.session_state.setdefault("refine_enabled", False)
    st.session_state.setdefault("refine_max_iterations", 3)
    st.session_state.setdefault("refine_threshold", 6.0)

    st.session_state.refine_enabled = st.toggle(
        "Habilitar refinamiento",
        value=st.session_state.refine_enabled,
        help="Permite refinar el juego después de la evaluación"
    )

    if st.session_state.refine_enabled:
        st.session_state.refine_max_iterations = st.slider(
            "Máx iteraciones",
            min_value=1,
            max_value=5,
            value=st.session_state.refine_max_iterations,
            help="Número máximo de ciclos de refinamiento"
        )

        st.session_state.refine_threshold = st.slider(
            "Umbral de calidad",
            min_value=4.0,
            max_value=8.0,
            value=st.session_state.refine_threshold,
            step=0.5,
            help="Puntuación mínima aceptable (el refinamiento se detiene al alcanzarla)"
        )

    st.markdown("---")

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
    st.text_input(
        "Estilo Artístico",
        placeholder="Ejemplo: fantasía oscura al óleo, anime vibrante, acuarela...",
        key="art_style",
        help="Define el estilo visual de las ilustraciones de las cartas"
    )

start_game = st.button("Iniciar Juego")

def call_api(endpoint: str, method: str = "POST", data: Dict[str, Any] = None, timeout: int = 300) -> Dict[str, Any]:
    try:
        url = f"http://localhost:8000/api/v1/games/{endpoint}"
        # Add premium query params
        premium = st.session_state.get("premium_mode", False)
        premium_provider = st.session_state.get("premium_provider", "gemini")
        separator = "&" if "?" in endpoint else "?"
        url += f"{separator}premium={str(premium).lower()}&premium_provider={premium_provider}"

        # Add model parameter for Gemini
        if premium and premium_provider == "gemini":
            gemini_model = st.session_state.get("gemini_model", "gemini-3.1-pro-preview")
            url += f"&model={gemini_model}"

        if method == "POST":
            response = requests.post(url, json=data, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
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
            "rule_complexity": st.session_state.get("rule_complexity") or None,
            "art_style": st.session_state.get("art_style") or None
        }

        refine_enabled = st.session_state.get("refine_enabled", False)
        max_iters = st.session_state.get("refine_max_iterations", 3)
        threshold = st.session_state.get("refine_threshold", 6.0)

        if refine_enabled:
            # Full automatic flow with progress
            with st.status("🚀 Generando juego completo...", expanded=True) as status:
                # Step 1: Start game
                st.write("📝 Iniciando juego...")
                result = call_api("start", data=data)
                if not result or "game_id" not in result:
                    st.error("Error al iniciar el juego")
                    st.stop()

                st.session_state.game_id = result["game_id"]
                game_id = st.session_state.game_id
                st.write(f"✅ Juego creado: `{game_id}`")

                # Step 2: Concept and Rules
                st.write("🎲 Generando concepto y reglas...")
                concept_result = call_api(f"{game_id}/concept-and-rules")
                if not concept_result or concept_result.get("status") != "rules_generated":
                    st.error("Error al generar concepto y reglas")
                    st.stop()
                st.write("✅ Concepto y reglas generados")

                # Step 3: Cards
                st.write("🃏 Generando cartas...")
                cards_result = call_api(f"{game_id}/cards", timeout=600)
                if not cards_result or cards_result.get("status") != "cards_generated":
                    st.error("Error al generar cartas")
                    st.stop()
                st.write("✅ Cartas generadas")

                # Step 4: Evaluate
                st.write("🤖 Evaluando juego...")
                eval_result = call_api(f"{game_id}/evaluate")
                if not eval_result:
                    st.error("Error al evaluar el juego")
                    st.stop()

                current_score = eval_result.get("overall_score", 0)
                st.write(f"📊 Puntuación inicial: **{current_score:.1f}/10**")

                # Step 5: Refine if needed (using refine-step for real-time updates)
                if current_score < threshold:
                    st.write(f"🔄 Refinando (objetivo: {threshold}, máx {max_iters} iteraciones)...")
                    iteration = 0
                    final_score = current_score
                    while iteration < max_iters and final_score < threshold:
                        iteration += 1
                        st.write(f"  ⏳ Iteración {iteration}/{max_iters}...")
                        step_result = call_api(
                            f"{game_id}/refine-step?threshold={threshold}",
                            timeout=600
                        )
                        if not step_result:
                            st.write(f"  ❌ Error en iteración {iteration}")
                            break
                        if step_result.get("status") == "threshold_met":
                            final_score = step_result.get("score", final_score)
                            st.write(f"  ✅ Umbral alcanzado: **{final_score:.1f}/10**")
                            break
                        prev = step_result.get("previous_score", 0)
                        new = step_result.get("new_score", 0)
                        improvement = step_result.get("improvement", 0)
                        sign = "+" if improvement > 0 else ""
                        st.write(f"  📊 {prev:.1f} → **{new:.1f}** ({sign}{improvement:.1f})")
                        final_score = new
                    if final_score >= threshold:
                        st.write(f"✅ ¡Refinamiento exitoso! Puntuación final: **{final_score:.1f}/10**")
                    else:
                        st.write(f"📈 Mejor puntuación alcanzada: **{final_score:.1f}/10** ({iteration} iteraciones)")
                else:
                    st.write(f"✅ ¡Puntuación ya cumple el umbral!")

                status.update(label="✅ ¡Juego completo!", state="complete", expanded=False)
                st.session_state.current_step = "complete"
                st.rerun()
        else:
            # Original manual flow
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
    else:
        st.session_state.game_id = game_id_input
        # Load game state to determine current step
        loaded_state = call_api(f"{game_id_input}", method="GET")
        if loaded_state:
            status = loaded_state.get("status", "")
            cards = loaded_state.get("cards") or []
            has_cards = bool(cards)
            all_have_images = has_cards and all(c.get("image_data") for c in cards)

            if status == "images_generated" or all_have_images:
                st.session_state.current_step = "complete"
            elif status == "cards_generated" or has_cards:
                st.session_state.current_step = "images"
            elif status == "rules_generated":
                st.session_state.current_step = "cards"
            else:
                st.session_state.current_step = "start"
            st.rerun()

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
                st.markdown(f"""🖼️ <b>Estilo Artístico:</b> {clean_and_unescape_text(concept.get("art_style", "No especificado"))}""", unsafe_allow_html=True)
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
                # Mapeo de las claves del modelo a títulos amigables para la interfaz (5 métricas con pesos)
                metric_map = {
                    "playability": ("Jugabilidad", 2.0),  # Most important
                    "balance": ("Balance", 1.5),
                    "clarity": ("Claridad", 1.2),
                    "theme_alignment": ("Alineación Temática", 1.0),
                    "innovation": ("Innovación", 0.8),
                }

                for key, (title, weight) in metric_map.items():
                    if key in evaluation:
                        metric_data = evaluation[key]
                        score = metric_data.get('adjusted_score') or metric_data.get('score', 'N/A')
                        score_display = f"{score:.1f}" if isinstance(score, float) else score
                        st.subheader(f"{title} (Puntuación: {score_display}/10, peso: {weight})")
                        st.markdown(f"**Análisis:** {metric_data.get('analysis', 'No hay análisis disponible.')}")
                        if metric_data.get("adjustment_reason"):
                            st.caption(f"📊 Ajuste: {metric_data.get('adjustment_reason')}")
                        if metric_data.get("suggestions"):
                            st.markdown("**Sugerencias de Mejora:**")
                            for suggestion in metric_data["suggestions"]:
                                st.markdown(f"- {suggestion}")
                        st.divider()

            # Show refinement button if enabled and score is below threshold
            if st.session_state.get("refine_enabled", False):
                current_score = evaluation.get('overall_score', 0)
                threshold = st.session_state.get("refine_threshold", 6.0)
                max_iters = st.session_state.get("refine_max_iterations", 3)

                if current_score < threshold:
                    st.warning(f"⚠️ Puntuación ({current_score:.1f}) por debajo del umbral ({threshold})")
                    if st.button("🔄 Refinar Juego"):
                        with st.status(f"🔄 Refinando (máx {max_iters} iteraciones)...", expanded=True) as refine_status:
                            iteration = 0
                            final_score = current_score
                            while iteration < max_iters and final_score < threshold:
                                iteration += 1
                                st.write(f"⏳ Iteración {iteration}/{max_iters}...")
                                step_result = call_api(
                                    f"{st.session_state.game_id}/refine-step?threshold={threshold}",
                                    timeout=600
                                )
                                if not step_result:
                                    st.write(f"❌ Error en iteración {iteration}")
                                    break
                                if step_result.get("status") == "threshold_met":
                                    final_score = step_result.get("score", final_score)
                                    st.write(f"✅ Umbral alcanzado: **{final_score:.1f}/10**")
                                    break
                                prev = step_result.get("previous_score", 0)
                                new = step_result.get("new_score", 0)
                                improvement = step_result.get("improvement", 0)
                                sign = "+" if improvement > 0 else ""
                                st.write(f"📊 {prev:.1f} → **{new:.1f}** ({sign}{improvement:.1f})")
                                final_score = new

                            if final_score >= threshold:
                                refine_status.update(label="✅ Refinamiento exitoso", state="complete")
                                st.success(f"Puntuación final: {final_score:.1f}/10")
                            else:
                                refine_status.update(label=f"📈 Mejor puntuación: {final_score:.1f}", state="complete")
                                st.info(f"No alcanzó el umbral ({threshold}) tras {iteration} iteraciones")
                        st.rerun()
                else:
                    st.success(f"✅ ¡El juego cumple el umbral de calidad! ({current_score:.1f} >= {threshold})")

        # Add evaluation button if game is ready for evaluation
        if game_state.get("status") in ["cards_generated", "images_generated"] and not game_state.get("evaluation"):
            if st.button("🤖 Evaluar Juego"):
                with st.spinner("Un experto en diseño de juegos está analizando tu creación..."):
                    result = call_api(f"{st.session_state.game_id}/evaluate")
                    if result and result.get("status") == "evaluated":
                        st.success("¡Evaluación completada! Los resultados se muestran arriba.")
                        st.rerun()

        # Show "Generar Cartas" button if we have rules but no cards
        if st.session_state.current_step == "cards" or (game_state.get("rules") and not game_state.get("cards")):
            if st.button("Generar Cartas"):
                with st.spinner("Generando cartas..."):
                    result = call_api(f"{st.session_state.game_id}/cards")
                    if result and result.get("status") == "cards_generated":
                        st.session_state.current_step = "images"
                        st.success("¡Cartas generadas! Ahora puedes generar las imágenes.")
                        st.rerun()

        # Show "Generar Imágenes" button or progress if any cards are missing images
        cards = game_state.get("cards") or []
        cards_without_images = [c for c in cards if not c.get("image_data")]

        # Check if generation is in progress
        status_result = call_api(f"{st.session_state.game_id}/images/status", method="GET", timeout=10)
        is_generating = status_result and status_result.get("generating", False)

        if is_generating:
            # Show progress
            completed = status_result.get("completed", 0)
            total = status_result.get("total", 0)
            progress = status_result.get("progress_percent", 0)
            remaining = status_result.get("remaining", 0)

            st.info(f"🎨 Generando imágenes en segundo plano... {completed}/{total} completadas")
            st.progress(progress / 100)
            st.caption(f"⏱️ Tiempo estimado restante: ~{remaining * 1.5:.0f} minutos")

            # Auto-refresh every 30 seconds
            time.sleep(3)
            st.rerun()
        elif cards_without_images:
            if st.button(f"🎨 Generar Imágenes ({len(cards_without_images)} pendientes)"):
                # Start background generation
                result = call_api(f"{st.session_state.game_id}/images", timeout=30)
                if result and result.get("status") == "generating":
                    st.success("¡Generación iniciada en segundo plano! Puedes cerrar esta página y volver más tarde.")
                    st.rerun()
