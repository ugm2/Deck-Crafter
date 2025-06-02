import streamlit as st
import requests
import json
from typing import Dict, Any

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

st.title("🎮 Deck Crafter")
st.markdown("Crea tu propio juego de cartas con IA")

if 'game_id' not in st.session_state:
    st.session_state.game_id = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'start'

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
        
        with st.spinner("Iniciando juego..."):
            result = call_api("start", data=data)
            if result and "game_id" in result:
                st.session_state.game_id = result["game_id"]
                st.session_state.current_step = "concept"
                game_state = call_api(f"{st.session_state.game_id}", method="GET")
                if game_state and "preferences" in game_state:
                    st.session_state["_pending_preferences"] = game_state["preferences"]
                    st.rerun()
                st.success("¡Juego iniciado! Ahora puedes generar el concepto.")
            else:
                st.session_state.game_id = None
                st.session_state.current_step = 'start'

if st.session_state.get("_just_updated_preferences"):
    st.session_state.current_step = "concept"
    st.success("¡Juego iniciado! Ahora puedes generar el concepto.")
    del st.session_state["_just_updated_preferences"]

if st.session_state.game_id:
    st.header("Estado del Juego")
    
    game_state = call_api(f"{st.session_state.game_id}", method="GET")
    if game_state:
        st.json(game_state)
    
    if st.session_state.current_step == "concept":
        if st.button("Generar Concepto"):
            with st.spinner("Generando concepto..."):
                result = call_api(f"{st.session_state.game_id}/concept")
                if result and result.get("status") == "concept_generated":
                    st.session_state.current_step = "rules"
                    st.success("¡Concepto generado! Ahora puedes generar las reglas.")
                    st.rerun()
    
    elif st.session_state.current_step == "rules":
        if st.button("Generar Reglas"):
            with st.spinner("Generando reglas..."):
                result = call_api(f"{st.session_state.game_id}/rules")
                if result and result.get("status") == "rules_generated":
                    st.session_state.current_step = "cards"
                    st.success("¡Reglas generadas! Ahora puedes generar las cartas.")
                    st.rerun()
    
    elif st.session_state.current_step == "cards":
        if st.button("Generar Cartas"):
            with st.spinner("Generando cartas..."):
                result = call_api(f"{st.session_state.game_id}/cards")
                if result and result.get("status") == "cards_generated":
                    st.session_state.current_step = "complete"
                    st.success("¡Juego completado! Puedes ver el resultado completo arriba.")
                    st.rerun()
