import streamlit as st
import requests

API_URL = "http://127.0.0.1:8001"

st.title("🔐 Auth Demo")

menu = st.sidebar.selectbox("Menu", ["Register", "Login", "Patients"])

# ── REGISTER ──
if menu == "Register":
    st.header("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        res = requests.post(f"{API_URL}/register", params={
            "username": username,
            "password": password
        })
        if res.status_code == 200:
            st.success("Registered! Now go login.")
        else:
            st.error(res.text)

# ── LOGIN ──
elif menu == "Login":
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        res = requests.post(f"{API_URL}/login", data={
            "username": username,
            "password": password
        })
        if res.status_code == 200:
            st.session_state["token"] = res.json()["access_token"]
            st.success("Logged in! ✅")
        else:
            st.error("Wrong credentials")

# ── PATIENTS ──
elif menu == "Patients":
    st.header("Patients")

    if "token" not in st.session_state:
        st.warning("⚠️ Login first!")
    else:
        res = requests.get(f"{API_URL}/patients", headers={
            "Authorization": f"Bearer {st.session_state['token']}"
        })

        if res.status_code == 200:
            st.success(f"Welcome **{res.json()['user']}**")
            st.json(res.json())
        elif res.status_code == 401:
            st.error("Token expired — login again")
        else:
            st.error(res.text)