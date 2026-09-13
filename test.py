import streamlit as st
import requests
import base64
import os

from rag import answer_query, load_vector_store
from chromaDB import build_vector_store, chroma_client

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Doctor Assistant", page_icon="🏥", layout="wide")
st.title("🏥 AI Doctor Assistant — Patient Management")

menu = st.sidebar.selectbox("Menu", [
    "➕ Add Patient",
    "📋 View All Patients",
    "🔍 Search by UID",
    "✏️ Update Patient",
    "🗑️ Delete Patient",
])


# ── ADD PATIENT ───────────────────────────────────────────────────────────────

if menu == "➕ Add Patient":
    st.header("Add New Patient")

    if "num_reports" not in st.session_state:
        st.session_state.num_reports = 1

    with st.form("add_patient_form"):
        col1, col2 = st.columns(2)
        with col1:
            name        = st.text_input("Full Name *")
            age         = st.number_input("Age *", min_value=0, max_value=150, step=1)
            gender      = st.selectbox("Gender *", ["Male", "Female", "Other"])
        with col2:
            blood_group = st.selectbox("Blood Group", ["", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            contact     = st.text_input("Contact Number")

        medical_history = st.text_area("Medical History / Notes")

        st.divider()
        st.subheader("Reports")
        for i in range(st.session_state.num_reports):
            st.text_input(f"Report Type {i+1}", key=f"report_type_{i}")
            st.file_uploader(f"Upload Report {i+1}", key=f"report_file_{i}")

        add_report = st.form_submit_button("➕ Add Another Report")
        submitted  = st.form_submit_button("Register Patient", type="primary")

    if add_report:
        st.session_state.num_reports += 1
        st.rerun()

    if submitted:
        if not name:
            st.error("Name is required.")
        else:
            reports = []
            for i in range(st.session_state.num_reports):
                report_type = st.session_state.get(f"report_type_{i}", "")
                report_file = st.session_state.get(f"report_file_{i}")
                if report_type and report_file:
                    encoded = base64.b64encode(report_file.read()).decode("utf-8")
                    reports.append({
                        "report_type": report_type,
                        "file_name":   report_file.name,
                        "file_data":   encoded,
                    })

            payload = {
                "name":            name,
                "age":             age,
                "gender":          gender,
                "blood_group":     blood_group or None,
                "contact":         contact or None,
                "medical_history": medical_history or None,
                "reports":         reports
            }
            res = requests.post(f"{API_URL}/patients", json=payload)
            if res.status_code == 201:
                data = res.json()
                st.success("Patient registered successfully!")
                st.info(f"**UID:** `{data['patient']['uid']}` — save this to look up the patient later.")
                st.json(data["patient"])
                st.session_state.num_reports = 1
            else:
                st.error(f"Error: {res.text}")


# ── VIEW ALL ──────────────────────────────────────────────────────────────────

elif menu == "📋 View All Patients":
    st.header("All Patients")
    res = requests.get(f"{API_URL}/patients")

    if res.status_code == 200:
        data = res.json()
        st.caption(f"Total patients: {data['total']}")

        if data["total"] == 0:
            st.info("No patients registered yet.")
        else:
            for p in data["patients"]:
                with st.expander(f"**{p['name']}** — `{p['uid']}`"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Age:** {p['age']}")
                        st.write(f"**Gender:** {p['gender']}")
                        st.write(f"**Blood Group:** {p.get('blood_group') or '—'}")
                    with col2:
                        st.write(f"**Contact:** {p.get('contact') or '—'}")
                        st.write(f"**Registered:** {p['created_at'][:10]}")

                    if p.get("medical_history"):
                        st.write(f"**Medical History:** {p['medical_history']}")

                    if p.get("reports"):
                        st.subheader("Reports")
                        for report in p["reports"]:
                            st.markdown(f"**{report['report_type']}** — `{report['file_name']}`")
                            file_bytes = base64.b64decode(report["file_data"])
                            if report["file_name"].lower().endswith((".png", ".jpg", ".jpeg")):
                                st.image(file_bytes, caption=f"{report['report_type']} report")
    else:
        st.error("Could not reach the API.")


# ── SEARCH BY UID ─────────────────────────────────────────────────────────────

elif menu == "🔍 Search by UID":
    st.header("Search Patient by UID")
    uid = st.text_input("Enter Patient UID (e.g. PAT-A1B2C3D4)")

    if st.button("Search", type="primary") and uid:
        res = requests.get(f"{API_URL}/patients/{uid.strip()}")
        if res.status_code == 200:
            st.session_state['searched_patient'] = res.json()
            st.session_state['searched_uid']     = uid.strip()
        elif res.status_code == 404:
            st.error("No patient found with that UID.")
        else:
            st.error(f"Error: {res.text}")

    if 'searched_patient' in st.session_state:
        p   = st.session_state['searched_patient']
        uid = st.session_state['searched_uid']

        st.success("Patient found!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Name",   p["name"])
            st.metric("Age",    p["age"])
            st.metric("Gender", p["gender"])
        with col2:
            st.metric("Blood Group", p.get("blood_group") or "—")
            st.metric("Contact",     p.get("contact") or "—")

        if p.get("medical_history"):
            st.subheader("Medical History")
            st.write(p["medical_history"])

        # ── Vector Store — check ChromaDB first, OCR only if needed ──────
        if st.session_state.get('current_uid') != uid or 'vector_store' not in st.session_state:

            vector_store = load_vector_store(uid)   # ✅ try loading from disk first

            if vector_store is not None:
                st.success("✅ Reports loaded instantly from cache!")
            else:
                # first time this patient is searched — run OCR once
                if p.get("reports"):
                    with st.spinner("Processing reports for the first time..."):
                        try:
                            vector_store = build_vector_store(p["reports"], uid)
                            st.success("✅ Reports processed and saved!")
                        except Exception as e:
                            st.warning(f"⚠️ Could not process reports: {e}")
                            vector_store = None
                else:
                    vector_store = None

            st.session_state['vector_store'] = vector_store

        vector_store = st.session_state.get('vector_store')

        # ── Always show reports ───────────────────────────────────────────
        if p.get("reports"):
            st.subheader("Reports")
            for report in p["reports"]:
                st.markdown(f"**{report['report_type']}** — `{report['file_name']}`")
                file_bytes = base64.b64decode(report["file_data"])
                if report["file_name"].lower().endswith((".png", ".jpg", ".jpeg")):
                    st.image(file_bytes, caption=f"{report['report_type']} report")

        # ── Load chat history if switching patient ────────────────────────
        if st.session_state.get('current_uid') != uid:
            chat_res = requests.get(f"{API_URL}/patients/{uid}/chat")
            if chat_res.status_code == 200:
                st.session_state['message_history'] = chat_res.json().get("messages", [])
            else:
                st.session_state['message_history'] = []
            st.session_state['current_uid'] = uid

        # ── Chat ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("🤖 Ask AI About This Patient")

        if 'message_history' not in st.session_state:
            st.session_state['message_history'] = []

        for message in st.session_state['message_history']:
            with st.chat_message(message['role']):
                st.text(message['content'])

        user_input = st.chat_input("Type here")
        if user_input:
            st.session_state['message_history'].append({'role': 'user', 'content': user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # save user message to backend
            requests.post(f"{API_URL}/patients/{uid}/chat", json={"role": "user", "content": user_input})

            # answer_query handles all RAG internally
            with st.chat_message("assistant"):
                ai_reply = st.write_stream(
                    chunk.content for chunk in answer_query(
                        vector_store, p, user_input, st.session_state['message_history']
                    )
                )

            # save AI reply to backend
            requests.post(f"{API_URL}/patients/{uid}/chat", json={"role": "assistant", "content": ai_reply})
            st.session_state['message_history'].append({'role': 'assistant', 'content': ai_reply})


# ── UPDATE PATIENT ────────────────────────────────────────────────────────────

elif menu == "✏️ Update Patient":
    st.header("Update Patient Details")
    uid = st.text_input("Enter Patient UID to update")

    if uid:
        res = requests.get(f"{API_URL}/patients/{uid.strip()}")
        if res.status_code == 200:
            p = res.json()
            st.success(f"Editing: **{p['name']}**")

            if "update_num_reports" not in st.session_state:
                st.session_state.update_num_reports = 1

            with st.form("update_form"):
                col1, col2 = st.columns(2)
                with col1:
                    new_name    = st.text_input("Name", value=p["name"])
                    new_age     = st.number_input("Age", value=p["age"], min_value=0, max_value=150)
                    new_gender  = st.selectbox("Gender", ["Male", "Female", "Other"],
                                               index=["Male", "Female", "Other"].index(p["gender"]))
                with col2:
                    new_bg      = st.text_input("Blood Group", value=p.get("blood_group") or "")
                    new_contact = st.text_input("Contact",     value=p.get("contact") or "")

                new_history = st.text_area("Medical History", value=p.get("medical_history") or "")
                st.divider()

                if p.get("reports"):
                    st.subheader("Existing Reports")
                    for report in p["reports"]:
                        st.markdown(f"**{report['report_type']}** — `{report['file_name']}`")
                        file_bytes = base64.b64decode(report["file_data"])
                        if report["file_name"].lower().endswith((".png", ".jpg", ".jpeg")):
                            st.image(file_bytes, caption=f"{report['report_type']} report")

                st.divider()
                st.subheader("Add New Reports")
                for i in range(st.session_state.update_num_reports):
                    st.text_input(f"New Report Type {i+1}",  key=f"upd_report_type_{i}")
                    st.file_uploader(f"Upload New Report {i+1}", key=f"upd_report_file_{i}")

                add_report = st.form_submit_button("➕ Add Another Report Field")
                submitted  = st.form_submit_button("Save Changes", type="primary")

            if add_report:
                st.session_state.update_num_reports += 1
                st.rerun()

            if submitted:
                new_reports = []
                for i in range(st.session_state.update_num_reports):
                    r_type = st.session_state.get(f"upd_report_type_{i}", "")
                    r_file = st.session_state.get(f"upd_report_file_{i}")
                    if r_type and r_file:
                        encoded = base64.b64encode(r_file.read()).decode("utf-8")
                        new_reports.append({
                            "report_type": r_type,
                            "file_name":   r_file.name,
                            "file_data":   encoded,
                        })

                payload = {
                    "name":            new_name,
                    "age":             new_age,
                    "gender":          new_gender,
                    "blood_group":     new_bg or None,
                    "contact":         new_contact or None,
                    "medical_history": new_history or None,
                }
                if new_reports:
                    payload["reports"] = new_reports

                res2 = requests.put(f"{API_URL}/patients/{uid.strip()}", json=payload)
                if res2.status_code == 200:
                    st.success("Patient updated successfully!")
                    st.session_state.update_num_reports = 1
                    st.json(res2.json()["patient"])
                else:
                    st.error(f"Error: {res2.text}")
        elif res.status_code == 404:
            st.warning("No patient found with that UID.")


# ── DELETE PATIENT ────────────────────────────────────────────────────────────

elif menu == "🗑️ Delete Patient":
    st.header("Delete Patient")
    uid = st.text_input("Enter Patient UID to delete")

    if uid:
        res = requests.get(f"{API_URL}/patients/{uid.strip()}")
        if res.status_code == 200:
            p = res.json()
            st.warning(f"You are about to delete **{p['name']}** (`{p['uid']}`). This cannot be undone.")
            if st.button("Confirm Delete", type="primary"):
                del_res = requests.delete(f"{API_URL}/patients/{uid.strip()}")
                if del_res.status_code == 200:
                    st.success(del_res.json()["message"])
                else:
                    st.error(f"Error: {del_res.text}")
        elif res.status_code == 404:
            st.warning("No patient found with that UID.")