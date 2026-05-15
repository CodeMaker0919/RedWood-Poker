import streamlit as st
import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import get_device
import torch

# 1. PAGE CONFIG
st.set_page_config(page_title="RedWood Bot", page_icon="🃏", layout="centered")
st.title("🃏 Play RedWood Bot")

# 2. INITIALIZE SESSION STATE
if 'env' not in st.session_state:
    st.session_state.env = rlcard.make('no-limit-holdem', config={'seed': 42})
    st.session_state.device = get_device()
    
    # LOAD BRAIN: Use weights_only=False to bypass the PyTorch security block
    checkpoint = torch.load('RedWood_bot_master.pth', map_location=st.session_state.device, weights_only=False)
    st.session_state.bot = DQNAgent.from_checkpoint(checkpoint=checkpoint)
    
    # We use a dummy for seat 0, Streamlit inputs will drive seat 0
    from rlcard.agents.random_agent import RandomAgent
    st.session_state.env.set_agents([RandomAgent(num_actions=st.session_state.env.num_actions), st.session_state.bot])
    
    st.session_state.game_over = True
    st.session_state.history = []

# 3. GAME ENGINE
def start_new_hand():
    state, player_id = st.session_state.env.reset()
    st.session_state.game_over = False
    st.session_state.history = ["New hand dealt."]
    process_turn(state, player_id)

def process_turn(state, player_id):
    while player_id == 1 and not st.session_state.env.is_over():
        action = st.session_state.bot.step(state)
        
        # Safe action naming
        action_name = f"Action {action}"
        if 'raw_legal_actions' in state:
            rla = state['raw_legal_actions']
            if isinstance(rla, dict):
                action_name = rla.get(action, action_name)
            elif isinstance(rla, list) and action < len(rla):
                action_name = rla[action]
                
        st.session_state.history.insert(0, f"🤖 RedWood chose: {action_name}")
        state, player_id = st.session_state.env.step(action)
    
    st.session_state.current_state = state
    st.session_state.player_id = player_id
    
    if st.session_state.env.is_over():
        st.session_state.game_over = True
        payoffs = st.session_state.env.get_payoffs()
        st.session_state.history.insert(0, f"🏁 GAME OVER! Your Payoff: {payoffs[0]} | RedWood: {payoffs[1]}")

def take_human_action(action_id, action_name):
    st.session_state.history.insert(0, f"👤 You chose: {action_name}")
    state, player_id = st.session_state.env.step(action_id)
    process_turn(state, player_id)

# 4. USER INTERFACE
if st.session_state.game_over:
    if st.button("Deal New Hand 🃏", use_container_width=True, type="primary"):
        start_new_hand()
        st.rerun()
else:
    raw_obs = st.session_state.current_state['raw_obs']
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Your Hand")
        st.info(" ".join(raw_obs['hand']))
    with c2:
        st.subheader("Community")
        if raw_obs['public_cards']:
            st.success(" ".join(raw_obs['public_cards']))
        else:
            st.warning("Pre-Flop")
            
    st.write(f"**Pot:** {raw_obs['pot']} chips")
    st.markdown("### Your Move")
    
    legal_actions = st.session_state.current_state['legal_actions']
    raw_legal_actions = st.session_state.current_state['raw_legal_actions']
    
    cols = st.columns(len(legal_actions))
    for i, action_id in enumerate(legal_actions):
        if isinstance(raw_legal_actions, dict):
            btn_label = raw_legal_actions.get(action_id, f"Action {action_id}")
        elif isinstance(raw_legal_actions, list) and i < len(raw_legal_actions):
            btn_label = raw_legal_actions[i]
        else:
            btn_label = f"Action {action_id}"
            
        with cols[i]:
            if st.button(str(btn_label).capitalize(), key=f"btn_{action_id}"):
                take_human_action(action_id, btn_label)
                st.rerun()

st.markdown("---")
st.markdown("### Match History")
for log in st.session_state.history:
    st.text(log)
