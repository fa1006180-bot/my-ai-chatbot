import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from services.gemini_service import generate_ai_image, stream_chat_response
from services.storage import (
    create_session,
    delete_session,
    list_sessions,
    load_session,
    save_image_bytes,
    save_session,
)

load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key, "").strip()
    if val:
        return val
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return default


st.set_page_config(
    page_title="Fardin AI Technology",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 5rem;
        max-width: 900px;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
    }
    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    sessions = list_sessions()
    if "current_session_id" not in st.session_state:
        if sessions:
            st.session_state.current_session_id = sessions[0]["id"]
        else:
            new_sess = create_session()
            st.session_state.current_session_id = new_sess["id"]


init_state()

# ---------------------- Sidebar (Chat History & Settings) ----------------------
with st.sidebar:
    st.title("🤖 Fardin AI Studio")
    st.caption("Chat • Image Analysis • AI Image Generation")

    if st.button("➕ New Chat", type="primary", use_container_width=True):
        new_sess = create_session()
        st.session_state.current_session_id = new_sess["id"]
        st.rerun()

    st.divider()
    st.subheader("🕒 Chat History")

    all_sessions = list_sessions()
    for sess in all_sessions:
        col1, col2 = st.columns([0.82, 0.18])
        is_active = sess["id"] == st.session_state.current_session_id
        label = f"💬 {sess['title']}"
        if len(label) > 26:
            label = label[:24] + "..."

        with col1:
            if st.button(
                label,
                key=f"select_{sess['id']}",
                type="secondary" if not is_active else "primary",
                use_container_width=True,
            ):
                st.session_state.current_session_id = sess["id"]
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{sess['id']}", help="Delete chat"):
                delete_session(sess["id"])
                remaining = list_sessions()
                if remaining:
                    st.session_state.current_session_id = remaining[0]["id"]
                else:
                    fresh = create_session()
                    st.session_state.current_session_id = fresh["id"]
                st.rerun()

    st.divider()
    with st.expander("⚙️ Model & API Settings", expanded=True):
        env_key = get_secret("GEMINI_API_KEY")
        api_key_input = st.text_input(
            "Gemini API Key (Chat & Vision)",
            value=env_key,
            type="password",
            help="Set GEMINI_API_KEY in Streamlit Secrets or .env file.",
        )
        if env_key:
            st.caption("✅ Server API Key loaded automatically.")

        chat_model = st.selectbox(
            "Chat & Vision Model",
            ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-4-hydrogen"],
            index=0,
            help="Use gemini-3.5-flash-lite for higher free daily quota (500 requests/day).",
        )

        st.markdown("---")
        image_provider = st.selectbox(
            "Image Generator Provider",
            [
                "Pollinations.ai (Free)",
                "Hugging Face (Free FLUX.1)",
                "Gemini (Paid Billing)",
            ],
            index=0,
        )

        hf_token = ""
        pollinations_key = ""
        gemini_image_model = "gemini-nano-banana-2.1"

        if image_provider == "Pollinations.ai (Free)":
            pollinations_key = st.text_input(
                "Pollinations Key (Optional)",
                value=get_secret("POLLINATIONS_API_KEY"),
                type="password",
                help="Leave blank to try anonymous mode, or get a free key at enter.pollinations.ai.",
            )
        elif image_provider == "Hugging Face (Free FLUX.1)":
            hf_token = st.text_input(
                "Hugging Face Token (Free)",
                value=get_secret("HF_TOKEN"),
                type="password",
                help="Create a free Read token at huggingface.co/settings/tokens (no card required).",
            )
        else:
            gemini_image_model = st.selectbox(
                "Gemini Image Model",
                ["gemini-nano-banana-2.1", "gemini-3.1-flash-lite-image"],
                index=0,
            )

api_key = api_key_input.strip()

current_session = load_session(st.session_state.current_session_id)
if current_session is None:
    current_session = create_session()
    st.session_state.current_session_id = current_session["id"]

# ---------------------- Main Chat Interface ----------------------
st.markdown(f"### {current_session['title']}")

mode = st.radio(
    "Select Mode:",
    options=["💬 Chat & Image Analysis", "🎨 Generate Image"],
    horizontal=True,
)

if not current_session["messages"]:
    st.info(
        "Welcome to **Fardin AI Technology**! Type any question below, or open **'📎 Upload an Image'** to get a detailed description of a photo. "
        "To create AI art, select **'🎨 Generate Image'** above (or start your message with `/image`)."
    )

for idx, msg in enumerate(current_session["messages"]):
    with st.chat_message(msg["role"]):
        if msg.get("image_path") and Path(msg["image_path"]).exists():
            st.image(msg["image_path"], width=380)
            if msg["role"] == "assistant":
                with open(msg["image_path"], "rb") as img_file:
                    st.download_button(
                        label="⬇️ Download Image",
                        data=img_file.read(),
                        file_name=Path(msg["image_path"]).name,
                        mime="image/png",
                        key=f"dl_{current_session['id']}_{idx}",
                    )
        if msg.get("text"):
            st.markdown(msg["text"])

# ---------------------- Image Upload Section ----------------------
with st.expander("📎 Upload an Image (for description or editing)", expanded=False):
    uploaded_file = st.file_uploader(
        "Choose a JPG, PNG, or WEBP image",
        type=["png", "jpg", "jpeg", "webp"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=260)
        analyze_now = st.button("🔍 Describe This Image", use_container_width=True)
    else:
        analyze_now = False

placeholder_text = (
    "Describe the image you want to generate..."
    if mode == "🎨 Generate Image"
    else "Ask a question (or type /image <prompt> to generate an image)..."
)
user_prompt = st.chat_input(placeholder_text)

if analyze_now and uploaded_file is not None:
    user_prompt = "Please describe what is in this image in detail."

if user_prompt or (uploaded_file is not None and analyze_now):
    prompt_text = (user_prompt or "").strip()

    is_image_gen = mode == "🎨 Generate Image"
    if prompt_text.lower().startswith("/image "):
        is_image_gen = True
        prompt_text = prompt_text[7:].strip()

    if (not is_image_gen or image_provider == "Gemini (Paid Billing)") and (
        not api_key or "আপনার" in api_key
    ):
        st.error("❌ Please enter a valid `GEMINI_API_KEY` in Streamlit Secrets or in the sidebar settings.")
        st.stop()

    saved_upload_path = None
    if uploaded_file is not None:
        saved_upload_path = save_image_bytes(uploaded_file.getvalue(), prefix="upload")
        st.session_state.uploader_key += 1

    if len(current_session["messages"]) == 0 and prompt_text:
        short_title = prompt_text[:30] + ("..." if len(prompt_text) > 30 else "")
        current_session["title"] = short_title

    with st.chat_message("user"):
        if saved_upload_path:
            st.image(saved_upload_path, width=380)
        if prompt_text:
            st.markdown(prompt_text)

    current_session["messages"].append(
        {
            "role": "user",
            "text": prompt_text,
            "image_path": saved_upload_path,
            "mode": "image_gen" if is_image_gen else "chat",
        }
    )

    with st.chat_message("assistant"):
        try:
            if is_image_gen:
                with st.spinner("🎨 Generating your image..."):
                    reply_text, gen_bytes = generate_ai_image(
                        provider=image_provider,
                        gemini_api_key=api_key,
                        chat_model=chat_model,
                        prompt=prompt_text,
                        reference_image_path=saved_upload_path,
                        hf_token=hf_token,
                        pollinations_key=pollinations_key,
                        gemini_image_model=gemini_image_model,
                    )

                saved_gen_path = None
                if gen_bytes:
                    saved_gen_path = save_image_bytes(gen_bytes, prefix="ai_gen")
                    st.image(saved_gen_path, width=420)
                    st.download_button(
                        label="⬇️ Download Image",
                        data=gen_bytes,
                        file_name=Path(saved_gen_path).name,
                        mime="image/png",
                    )
                if reply_text:
                    st.markdown(reply_text)
                elif not gen_bytes:
                    reply_text = "Sorry, could not generate an image from this prompt."
                    st.warning(reply_text)

                current_session["messages"].append(
                    {
                        "role": "assistant",
                        "text": reply_text,
                        "image_path": saved_gen_path,
                        "mode": "image_gen",
                    }
                )
            else:
                history_for_api = current_session["messages"][:-1]
                full_response = st.write_stream(
                    stream_chat_response(
                        api_key=api_key,
                        model_name=chat_model,
                        history_messages=history_for_api,
                        user_text=prompt_text,
                        image_path=saved_upload_path,
                    )
                )
                current_session["messages"].append(
                    {
                        "role": "assistant",
                        "text": full_response,
                        "image_path": None,
                        "mode": "chat",
                    }
                )

            save_session(current_session)
            st.rerun()

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
