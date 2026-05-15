import streamlit as st
import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import get_device

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="RedWood Bot vs Human", page_icon="🃏", layout="centered")
st.title("🃏 Play RedWood Bot (3M IQ)")

# --- 2. INITIALIZE GAME STATE ---
# We use st.session_state so the app remembers the game between button clicks
if 'env' not in st.session_state:
    st.session_state.env = rlcard.make('no-limit-holdem', config={'seed': 42})
    st.session_state.device = get_device()
    
    # Load the Bot
    st.session_state.bot = DQNAgent(
        num_actions=st.session_state.env.num_actions,
        state_shape=st.session_state.env.state_shape[0],
        mlp_layers=[128, 128],
        device=st.session_state.device
    )
    st.session_state.bot.load_checkpoint(path='.', filename='RedWood_bot_master.pth')    
    # We don't need a HumanAgent here, Streamlit IS the human agent.
    # We put a dummy agent in Seat 0 just to satisfy the env, but we will manually pass human actions.
    from rlcard.agents.random_agent import RandomAgent
    st.session_state.env.set_agents([RandomAgent(num_actions=st.session_state.env.num_actions), st.session_state.bot])
    
    st.session_state.game_over = True
    st.session_state.history = []

# --- 3. GAME LOGIC FUNCTIONS ---
def start_new_hand():
    state, player_id = st.session_state.env.reset()
    st.session_state.game_over = False
    st.session_state.history = ["New hand dealt."]
    process_turn(state, player_id)

def process_turn(state, player_id):
    # If it's the bot's turn (Player 1), let it play automatically until it's the human's turn
    while player_id == 1 and not st.session_state.env.is_over():
        action = st.session_state.bot.step(state)
        action_name = state['raw_legal_actions'][action] if 'raw_legal_actions' in state else f"Action {action}"
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

# --- 4. THE UI INTERFACE ---
if st.session_state.game_over:
    if st.button("Deal New Hand 🃏", use_container_width=True, type="primary"):
        start_new_hand()
        st.rerun()

else:
    # Get the human-readable state
    raw_obs = st.session_state.current_state['raw_obs']
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Hand")
        st.info(" ".join(raw_obs['hand']))
    with col2:
        st.subheader("Community Cards")
        if raw_obs['public_cards']:
            st.success(" ".join(raw_obs['public_cards']))
        else:
            st.warning("Pre-Flop (No cards yet)")
            
    st.write(f"**Pot Size:** {raw_obs['pot']} chips")

    st.markdown("### Your Move")
    # Dynamically generate buttons based on what moves are legally allowed right now
    legal_actions = st.session_state.current_state['legal_actions']
    raw_legal_actions = st.session_state.current_state['raw_legal_actions']
    
    # Create a row of buttons for each legal action
    cols = st.columns(len(legal_actions))
    for i, action_id in enumerate(legal_actions):
        action_name = raw_legal_actions.get(action_id, f"Action {action_id}")
        with cols[i]:
            if st.button(str(action_name).capitalize(), key=f"btn_{action_id}"):
                take_human_action(action_id, action_name)
                st.rerun()

# --- 5. GAME LOG ---
st.markdown("---")
st.markdown("### Match History")
for log in st.session_state.history:
    st.text(log)
