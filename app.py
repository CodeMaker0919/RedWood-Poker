import streamlit as st
import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import get_device
import torch

# --- CUSTOM THEME & CSS ---
st.set_page_config(page_title="RedWood Poker", page_icon="🃏", layout="centered")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; font-weight: bold; }
    .card-box { background-color: #1e1e1e; padding: 20px; border-radius: 15px; text-align: center; border: 2px solid #333; }
    .pot-box { background-color: #0e1117; padding: 10px; border-radius: 10px; text-align: center; color: #ffca28; font-size: 1.2em; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("⚔️ RedWood Poker Arena")

# --- INITIALIZATION ---
if 'env' not in st.session_state:
    st.session_state.env = rlcard.make('no-limit-holdem', config={'seed': 42})
    st.session_state.device = get_device()
    checkpoint = torch.load('RedWood_bot_master.pth', map_location=st.session_state.device, weights_only=False)
    st.session_state.bot = DQNAgent.from_checkpoint(checkpoint=checkpoint)
    from rlcard.agents.random_agent import RandomAgent
    st.session_state.env.set_agents([RandomAgent(num_actions=st.session_state.env.num_actions), st.session_state.bot])
    st.session_state.game_over = True
    st.session_state.history = []

# --- CORE LOGIC ---
def start_new_hand():
    state, player_id = st.session_state.env.reset()
    st.session_state.game_over = False
    st.session_state.history = ["📢 New hand dealt."]
    process_turn(state, player_id)

def process_turn(state, player_id):
    while player_id == 1 and not st.session_state.env.is_over():
        action = st.session_state.bot.step(state)
        rla = state.get('raw_legal_actions', [])
        action_name = rla[action] if isinstance(rla, list) and action < len(rla) else f"Action {action}"
        st.session_state.history.insert(0, f"🤖 RedWood chose: {action_name}")
        state, player_id = st.session_state.env.step(action)
    st.session_state.current_state, st.session_state.player_id = state, player_id
    if st.session_state.env.is_over():
        st.session_state.game_over = True
        p = st.session_state.env.get_payoffs()
        st.session_state.history.insert(0, f"🏁 Result -> You: {p[0]} | Bot: {p[1]}")

# --- UI LAYOUT ---
if st.session_state.game_over:
    st.markdown("<div class='pot-box'>Table Closed</div>", unsafe_allow_html=True)
    if st.button("Deal New Hand 🃏", type="primary"):
        start_new_hand()
        st.rerun()
else:
    raw_obs = st.session_state.current_state['raw_obs']
    
    # Community Cards
    st.markdown("### 🏛️ Community Board")
    board = " ".join(raw_obs['public_cards']) if raw_obs['public_cards'] else "🎴 🎴 🎴 🎴 🎴"
    st.markdown(f"<div class='card-box' style='font-size: 2em; color: #4caf50;'>{board}</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='pot-box'>💰 Pot: {raw_obs['pot']} Chips</div>", unsafe_allow_html=True)

    # Player Hand
    st.markdown("### 🤲 Your Hand")
    hand = " ".join(raw_obs['hand'])
    st.markdown(f"<div class='card-box' style='font-size: 2.5em; border-color: #2196f3;'>{hand}</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    # Action Buttons
    legal_actions = st.session_state.current_state['legal_actions']
    raw_legal_actions = st.session_state.current_state['raw_legal_actions']
    cols = st.columns(len(legal_actions))
    for i, action_id in enumerate(legal_actions):
        btn_label = raw_legal_actions[action_id] if isinstance(raw_legal_actions, dict) else raw_legal_actions[i]
        with cols[i]:
            if st.button(str(btn_label).upper(), key=f"btn_{action_id}"):
                st.session_state.history.insert(0, f"👤 You chose: {btn_label}")
                state, p_id = st.session_state.env.step(action_id)
                process_turn(state, p_id)
                st.rerun()

# History Log
with st.expander("📝 Hand Log", expanded=True):
    for log in st.session_state.history[:5]:
        st.write(log)
