#%%
import streamlit as st

PERSONAS = {
    "🧑‍⚕️ 상담사 지우": {
        "description": "따뜻하고 공감 잘 하는 전문 심리 상담사",
        "system": "당신은 '지우'라는 이름의 전문 심리 상담사입니다.",
        "color": "#4A90D9", "emoji": "🧑‍⚕️",
    },
    "👫 친구 민준": {
        "description": "편하게 털어놓을 수 있는 동네 친구",
        "system": "너는 '민준'이라는 이름의 오랜 친구야. 반말로 대화해.",
        "color": "#27AE60", "emoji": "👫",
    },
    "🤖 AI 어시스턴트 클로": {
        "description": "Claude/GPT 스타일의 지식형 어시스턴트",
        "system": "당신은 '클로'라는 AI 어시스턴트입니다.",
        "color": "#8E44AD", "emoji": "🤖",
    },
    "📚 멘토 선생님": {
        "description": "논리적이고 조언 잘 해주는 인생 멘토",
        "system": "당신은 경험이 풍부한 인생 멘토입니다.",
        "color": "#E67E22", "emoji": "📚",
    },
    "😄 개그맨 철수": {
        "description": "유머러스하고 긍정 에너지 넘치는 친구",
        "system": "너는 '철수'야. 유머 넘치는 친구야. 반말로 대화해.",
        "color": "#E74C3C", "emoji": "😄",
    },
}

st.set_page_config(page_title="페르소나 챗봇", page_icon="💬", layout="centered")
#%%
if "selected_persona" not in st.session_state:
    st.session_state.selected_persona = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("## 💬 페르소나 선택")
    st.markdown("대화 상대를 골라보세요!")
    st.divider()
    for name, info in PERSONAS.items():
        if st.button(f"{info['emoji']} {name.split(' ', 1)[1]}\n\n_{info['description']}_", key=name, use_container_width=True):
            if st.session_state.selected_persona != name:
                st.session_state.selected_persona = name
                st.session_state.messages = []
                st.rerun()
    st.divider()
    if st.session_state.selected_persona:
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
#%%
if st.session_state.selected_persona is None:
    st.markdown("# 💬 페르소나 챗봇")
    st.markdown("왼쪽 사이드바에서 대화 상대를 선택하세요!")
    cols = st.columns(2)
    for i, (name, info) in enumerate(PERSONAS.items()):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:{info['color']}22; border:2px solid {info['color']}44;
                        border-radius:12px; padding:12px; margin-bottom:8px;">
                <div style="font-size:2rem;">{info['emoji']}</div>
                <div style="font-weight:700; color:{info['color']};">{name.split(' ',1)[1]}</div>
                <div style="font-size:0.85rem; color:#666; margin-top:4px;">{info['description']}</div>
            </div>""", unsafe_allow_html=True)
else:
    persona = PERSONAS[st.session_state.selected_persona]
    st.markdown(f"""
    <div style="background:{persona['color']}; border-radius:12px; padding:16px 20px;
                margin-bottom:20px; color:white; font-weight:700;">
        {persona['emoji']} {st.session_state.selected_persona.split(' ',1)[1]}
        <span style="font-weight:400; font-size:0.9rem; margin-left:10px; opacity:0.85;">
            {persona['description']}
        </span>
    </div>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("메시지를 입력하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            # API 연결 전 테스트용 더미 응답
            reply = f"[테스트] '{prompt}' 라고 하셨군요! (실제 API 응답 자리입니다)"
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})