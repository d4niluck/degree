import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
import pandas as pd
import requests

from schemas import KnowledgeBaseCreateForm, UserCreateForm, UserKnowledgeBaseCreateForm


st.set_page_config(page_title="Doc Helper", layout="wide")
st.set_option("client.toolbarMode", "auto")

API_BASE_URL = "http://127.0.0.1:8000"
ADMIN_LOGIN = os.getenv("APP_ADMIN_LOGIN", "admin")
ADMIN_PASSWORD = os.getenv("APP_ADMIN_PASSWORD", "admin")
USER_LOGIN = os.getenv("APP_USER_LOGIN", "user")
USER_PASSWORD = os.getenv("APP_USER_PASSWORD", "user")
UPLOAD_DIR = Path(os.getenv("FRONTEND_UPLOAD_DIR", "data/frontend_uploads")).resolve()


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.get('token')}"}


def login_as(role: str) -> None:
    credentials = {
        "admin": {"login": ADMIN_LOGIN, "password": ADMIN_PASSWORD},
        "user": {"login": USER_LOGIN, "password": USER_PASSWORD},
    }[role]
    response = requests.post(f"{API_BASE_URL}/login", json=credentials, timeout=30)
    response.raise_for_status()
    token = response.json()["access_token"]
    st.session_state["token"] = token
    st.session_state["role"] = role


def invalidate_admin_tables() -> None:
    st.session_state.pop("users_df", None)
    st.session_state.pop("knowledge_bases_df", None)


def invalidate_user_data() -> None:
    st.session_state.pop("user_bases_df", None)
    for key in list(st.session_state):
        if str(key).startswith("user_documents_df_"):
            st.session_state.pop(key, None)


def fetch_users_df() -> pd.DataFrame:
    response = requests.get(
        f"{API_BASE_URL}/admin/users",
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json()
    if not rows:
        return pd.DataFrame(columns=["user_id", "login", "role", "is_active"])
    dataframe = pd.DataFrame(rows)
    columns = [column for column in ["user_id", "login", "role", "is_active"] if column in dataframe.columns]
    return dataframe[columns]


def fetch_knowledge_bases_df() -> pd.DataFrame:
    response = requests.get(
        f"{API_BASE_URL}/admin/knowledge-bases",
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json()
    columns = ["kb_id", "name", "owner_user_id", "owner_login", "storage_path", "created_at"]
    if not rows:
        return pd.DataFrame(columns=columns)
    dataframe = pd.DataFrame(rows)
    columns = [column for column in columns if column in dataframe.columns]
    return dataframe[columns]


def fetch_user_knowledge_bases_df() -> pd.DataFrame:
    response = requests.get(
        f"{API_BASE_URL}/knowledge-bases",
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json()
    columns = ["kb_id", "name", "owner_user_id", "storage_path"]
    if not rows:
        return pd.DataFrame(columns=columns)
    dataframe = pd.DataFrame(rows)
    columns = [column for column in columns if column in dataframe.columns]
    return dataframe[columns]


def create_user_knowledge_base(payload: UserKnowledgeBaseCreateForm) -> None:
    response = requests.post(
        f"{API_BASE_URL}/knowledge-bases",
        json=payload.model_dump(),
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    invalidate_user_data()


def delete_user_knowledge_base(kb_id: str) -> None:
    response = requests.delete(
        f"{API_BASE_URL}/knowledge-bases/{kb_id}",
        headers=auth_headers(),
        timeout=120,
    )
    response.raise_for_status()
    invalidate_user_data()


def fetch_documents_df(kb_id: str) -> pd.DataFrame:
    response = requests.get(
        f"{API_BASE_URL}/knowledge-bases/{kb_id}/documents",
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json()
    columns = ["doc_id", "name", "source_path", "pages"]
    if not rows:
        return pd.DataFrame(columns=columns)
    dataframe = pd.DataFrame(rows)
    columns = [column for column in columns if column in dataframe.columns]
    return dataframe[columns]


def add_document(kb_id: str, file_path: str) -> None:
    response = requests.post(
        f"{API_BASE_URL}/knowledge-bases/{kb_id}/documents",
        json={"file_path": file_path},
        headers=auth_headers(),
        timeout=600,
    )
    response.raise_for_status()
    st.session_state.pop(f"user_documents_df_{kb_id}", None)


def delete_document(kb_id: str, doc_id: str) -> None:
    response = requests.delete(
        f"{API_BASE_URL}/knowledge-bases/{kb_id}/documents",
        json={"doc_id": doc_id},
        headers=auth_headers(),
        timeout=120,
    )
    response.raise_for_status()
    st.session_state.pop(f"user_documents_df_{kb_id}", None)


def ask_question(kb_id: str, question: str, source_paths: list[str] | None) -> str:
    response = requests.post(
        f"{API_BASE_URL}/knowledge-bases/{kb_id}/ask",
        json={
            "question": question,
            "top_k": 10,
            "source_paths": source_paths,
        },
        headers=auth_headers(),
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["answer"]


def save_uploaded_file(uploaded_file) -> str:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    path = UPLOAD_DIR / f"{uuid4().hex}_{safe_name}"
    path.write_bytes(uploaded_file.getvalue())
    return str(path)


def create_user(payload: UserCreateForm) -> None:
    response = requests.post(
        f"{API_BASE_URL}/admin/users",
        json=payload.model_dump(),
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    invalidate_admin_tables()


def delete_user(user_id: str) -> None:
    response = requests.delete(
        f"{API_BASE_URL}/admin/users/{user_id}",
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    invalidate_admin_tables()


def create_knowledge_base(payload: KnowledgeBaseCreateForm) -> None:
    response = requests.post(
        f"{API_BASE_URL}/admin/knowledge-bases",
        json=payload.model_dump(),
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    invalidate_admin_tables()


def delete_knowledge_base(kb_id: str) -> None:
    response = requests.delete(
        f"{API_BASE_URL}/admin/knowledge-bases/{kb_id}",
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    invalidate_admin_tables()


def select_role(role: str) -> None:
    try:
        login_as(role)
    except Exception as exc:
        st.error(f"Не удалось войти как {role}: {exc}")


def reset_role() -> None:
    st.session_state.pop("role", None)
    st.session_state.pop("token", None)
    invalidate_admin_tables()
    invalidate_user_data()


def toggle_admin_edit_mode() -> None:
    st.session_state["admin_edit_mode"] = not st.session_state.get("admin_edit_mode", False)


def toggle_user_edit_mode() -> None:
    st.session_state["user_edit_mode"] = not st.session_state.get("user_edit_mode", False)


def render_role_picker() -> None:
    st.title("Выберите роль", text_alignment='center')
    _, left, right, _ = st.columns([1.5, 1, 1, 1.5])
    with left:
        st.button("Администратор", use_container_width=True, on_click=select_role, args=("admin",))
    with right:
        st.button("Пользователь", use_container_width=True, on_click=select_role, args=("user",))


def render_global_header() -> None:
    role = st.session_state.get("role")
    _, right = st.columns([8, 1])
    with right:
        st.button("Профиль", use_container_width=True, on_click=reset_role)

    if role == 'user':
        pass
    if role == 'admin':
        pass
    

def render_admin() -> None:
    render_global_header()
    st.title("Панель управления")

    if "admin_edit_mode" not in st.session_state:
        st.session_state["admin_edit_mode"] = False

    refresh_col, edit_col, _ = st.columns([1, 1.5, 5])
    with refresh_col:
        if st.button("Обновить", use_container_width=True):
            invalidate_admin_tables()
    with edit_col:
        st.button(
            "Закрыть редактирование" if st.session_state["admin_edit_mode"] else "Режим редактирования",
            use_container_width=True,
            on_click=toggle_admin_edit_mode,
        )

    edit_mode = st.session_state["admin_edit_mode"]
    col1, col2 = st.columns(2)
    with col1:
        render_users_panel(edit_mode)
    with col2:
        render_knowledge_bases_panel(edit_mode)


def render_users_panel(edit_mode: bool) -> None:
    st.subheader("Пользователи")
    if "users_df" not in st.session_state:
        try:
            st.session_state["users_df"] = fetch_users_df()
        except Exception as exc:
            st.error(f"Не удалось загрузить пользователей: {exc}")
            st.session_state["users_df"] = pd.DataFrame(
                columns=["user_id", "login", "role", "is_active"]
            )
    users_df = st.session_state["users_df"]
    if edit_mode:
        edited_users_df = users_df.copy()
        edited_users_df["delete"] = False
        edited_users_df = st.data_editor(
            edited_users_df,
            use_container_width=True,
            hide_index=True,
            disabled=["user_id", "login", "role", "is_active"],
            column_config={
                "delete": st.column_config.CheckboxColumn("Удалить"),
            },
        )
    else:
        st.dataframe(users_df, use_container_width=True, hide_index=True)
        return

    _, apply_col = st.columns([3, 1])
    with apply_col:
        if st.button("Применить удаление", key="apply_delete_users", use_container_width=True):
            rows_to_delete = edited_users_df[edited_users_df["delete"]]
            try:
                for _, row in rows_to_delete.iterrows():
                    delete_user(row["user_id"])
                st.success(f"Удалено пользователей: {len(rows_to_delete)}")
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось удалить пользователей: {exc}")

    st.divider()
    st.caption("Добавить пользователя")
    with st.form("create_user_form"):
        login = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")
        role = st.selectbox("Роль", ["user", "admin"])
        submitted = st.form_submit_button("Добавить пользователя", use_container_width=True)
    if submitted:
        try:
            create_user(UserCreateForm(login=login, password=password, role=role))
            st.session_state.pop("users_df", None)
            st.success("Пользователь добавлен")
            st.rerun()
        except Exception as exc:
            st.error(f"Не удалось добавить пользователя: {exc}")


def render_knowledge_bases_panel(edit_mode: bool) -> None:
    st.subheader("Базы знаний")
    if "knowledge_bases_df" not in st.session_state:
        try:
            st.session_state["knowledge_bases_df"] = fetch_knowledge_bases_df()
        except Exception as exc:
            st.error(f"Не удалось загрузить базы знаний: {exc}")
            st.session_state["knowledge_bases_df"] = pd.DataFrame(
                columns=["kb_id", "name", "owner_user_id", "owner_login", "storage_path", "created_at"]
            )
    kb_df = st.session_state["knowledge_bases_df"]
    if edit_mode:
        edited_kb_df = kb_df.copy()
        edited_kb_df["delete"] = False
        edited_kb_df = st.data_editor(
            edited_kb_df,
            use_container_width=True,
            hide_index=True,
            disabled=["kb_id", "name", "owner_user_id", "owner_login", "storage_path", "created_at"],
            column_config={
                "delete": st.column_config.CheckboxColumn("Удалить"),
            },
        )
    else:
        st.dataframe(kb_df, use_container_width=True, hide_index=True)
        return

    _, apply_col = st.columns([3, 1])
    with apply_col:
        if st.button("Применить удаление", key="apply_delete_kb", use_container_width=True):
            rows_to_delete = edited_kb_df[edited_kb_df["delete"]]
            try:
                for _, row in rows_to_delete.iterrows():
                    delete_knowledge_base(row["kb_id"])
                st.success(f"Удалено баз знаний: {len(rows_to_delete)}")
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось удалить базы знаний: {exc}")

    st.divider()
    st.caption("Добавить базу знаний")
    with st.form("create_kb_form"):
        owner_user_id = st.text_input("ID владельца")
        name = st.text_input("Название базы знаний")
        submitted = st.form_submit_button("Добавить базу знаний", use_container_width=True)
    if submitted:
        try:
            create_knowledge_base(KnowledgeBaseCreateForm(owner_user_id=owner_user_id, name=name))
            st.session_state.pop("knowledge_bases_df", None)
            st.success("База знаний добавлена")
            st.rerun()
        except Exception as exc:
            st.error(f"Не удалось добавить базу знаний: {exc}")


def render_user() -> None:
    render_global_header()
    st.title("Пользователь")

    if "user_edit_mode" not in st.session_state:
        st.session_state["user_edit_mode"] = False

    refresh_col, edit_col, _ = st.columns([1, 1.5, 5])
    with refresh_col:
        if st.button("Обновить", key="refresh_user_data", use_container_width=True):
            invalidate_user_data()
    with edit_col:
        st.button(
            "Закрыть редактирование" if st.session_state["user_edit_mode"] else "Режим редактирования",
            key="toggle_user_edit_mode",
            use_container_width=True,
            on_click=toggle_user_edit_mode,
        )

    kb_df = get_user_knowledge_bases_df()
    if kb_df.empty:
        try:
            create_user_knowledge_base(UserKnowledgeBaseCreateForm(name="База знаний 1"))
            kb_df = fetch_user_knowledge_bases_df()
            st.session_state["user_bases_df"] = kb_df
        except Exception as exc:
            st.error(f"Не удалось создать базу знаний: {exc}")
            return

    col1, col2 = st.columns([1.1, 0.9])
    with col2:
        st.subheader("База знаний")
        kb_id = render_user_kb_selector(kb_df)
        if not kb_id:
            return
        selected_source_paths = render_user_documents_panel(kb_id)
    with col1:
        st.subheader("Ассистент")
        render_assistant_chat(kb_id, selected_source_paths)


def get_user_knowledge_bases_df() -> pd.DataFrame:
    if "user_bases_df" not in st.session_state:
        try:
            st.session_state["user_bases_df"] = fetch_user_knowledge_bases_df()
        except Exception as exc:
            st.error(f"Не удалось загрузить базы знаний: {exc}")
            st.session_state["user_bases_df"] = pd.DataFrame(
                columns=["kb_id", "name", "owner_user_id", "storage_path"]
            )
    return st.session_state["user_bases_df"]


def render_user_kb_selector(kb_df: pd.DataFrame) -> str | None:
    select_col, create_col = st.columns([4, 1])
    names = kb_df["name"].tolist()
    with select_col:
        selected_name = st.selectbox("База знаний", names)
    with create_col:
        st.write("")
        st.write("")
        selected_row = kb_df[kb_df["name"] == selected_name]
        if selected_row.empty:
            return None
        kb_id = str(selected_row.iloc[0]["kb_id"])
        if st.button("Удалить", use_container_width=True):
            try:
                delete_user_knowledge_base(kb_id)
                st.success("База знаний удалена")
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось удалить базу знаний: {exc}")

    left, right = st.columns([3, 8])
    with left:
        if st.button("Добавить базу", use_container_width=True):
            st.session_state["show_user_create_kb_form"] = True


    if st.session_state.get("show_user_create_kb_form"):
        with st.form("user_create_kb_form"):
            name = st.text_input("Название новой базы знаний")
            create_col, cancel_col = st.columns(2)
            with create_col:
                submitted = st.form_submit_button("Создать", use_container_width=True)
            with cancel_col:
                cancelled = st.form_submit_button("Отмена", use_container_width=True)
        if cancelled:
            st.session_state["show_user_create_kb_form"] = False
            st.rerun()
        if submitted:
            try:
                create_user_knowledge_base(UserKnowledgeBaseCreateForm(name=name))
                st.session_state["show_user_create_kb_form"] = False
                st.success("База знаний создана")
                st.rerun()
            except Exception as exc:
                st.error(f"Не удалось создать базу знаний: {exc}")

    return kb_id


def render_assistant_chat(kb_id: str, selected_source_paths: list[str] | None) -> None:
    chat_state_key = f"last_chat_message_{kb_id}"
    last_message = st.session_state.get(chat_state_key)
    if last_message:
        with st.chat_message("user"):
            st.markdown(last_message["question"])
        with st.chat_message("assistant"):
            st.markdown(last_message["answer"])

    question = st.chat_input("Задайте вопрос по выбранным документам")
    if not question:
        return
    if not selected_source_paths:
        st.warning("Выберите хотя бы один документ для поиска")
        return

    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.chat_message("assistant"):
            with st.spinner("Формирую ответ"):
                answer = ask_question(kb_id, question, selected_source_paths)
            st.markdown(answer)
        st.session_state[chat_state_key] = {
            "question": question,
            "answer": answer,
        }
    except Exception as exc:
        st.error(f"Не удалось получить ответ: {exc}")


def render_user_documents_panel(kb_id: str) -> list[str] | None:
    st.subheader("Документы")

    doc_state_key = f"user_documents_df_{kb_id}"
    if st.session_state.get("selected_kb_id") != kb_id:
        st.session_state["selected_kb_id"] = kb_id

    if doc_state_key not in st.session_state:
        try:
            st.session_state[doc_state_key] = fetch_documents_df(kb_id)
        except Exception as exc:
            st.error(f"Не удалось загрузить документы: {exc}")
            st.session_state[doc_state_key] = pd.DataFrame(
                columns=["doc_id", "name", "source_path", "pages"]
            )

    documents_df = st.session_state[doc_state_key]
    edit_mode = st.session_state.get("user_edit_mode", False)
    if documents_df.empty:
        render_document_upload(kb_id)
        return None

    editor_df = documents_df.copy()
    editor_df["use"] = True
    if edit_mode:
        editor_df["delete"] = False

    disabled_columns = ["doc_id", "name", "source_path", "pages"]
    visible_columns = ["name", "pages", "use"]
    if edit_mode:
        visible_columns.append("delete")
    column_config = {
        "name": st.column_config.TextColumn("Файл"),
        "pages": st.column_config.NumberColumn("Страниц"),
        "use": st.column_config.CheckboxColumn("Использовать"),
    }
    if edit_mode:
        column_config["delete"] = st.column_config.CheckboxColumn("Удалить")

    edited_df = st.data_editor(
        editor_df,
        use_container_width=True,
        hide_index=True,
        column_order=visible_columns,
        disabled=disabled_columns,
        column_config=column_config,
    )

    if edit_mode:
        _, apply_col = st.columns([3, 1])
        with apply_col:
            if st.button("Применить удаление", key=f"apply_delete_docs_{kb_id}", use_container_width=True):
                rows_to_delete = edited_df[edited_df["delete"]]
                try:
                    for _, row in rows_to_delete.iterrows():
                        delete_document(kb_id, row["doc_id"])
                    st.session_state.pop(doc_state_key, None)
                    st.success(f"Удалено документов: {len(rows_to_delete)}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Не удалось удалить документы: {exc}")

    selected_paths = edited_df[edited_df["use"]]["source_path"].dropna().tolist()
    # TODO: вернуть preview PDF, когда будет выбран стабильный viewer для Streamlit.
    render_document_upload(kb_id)
    return selected_paths


def render_document_upload(kb_id: str) -> None:
    upload_key_state = f"upload_key_{kb_id}"
    if upload_key_state not in st.session_state:
        st.session_state[upload_key_state] = uuid4().hex

    uploader_key = f"document_uploader_{kb_id}_{st.session_state[upload_key_state]}"
    uploaded_file = st.file_uploader("Загрузить новый файл", type=["pdf"], key=uploader_key)
    if uploaded_file and st.button("Добавить документ", use_container_width=True):
        try:
            file_path = save_uploaded_file(uploaded_file)
            with st.spinner("Добавляю документ в индекс"):
                add_document(kb_id, file_path)
            st.session_state[upload_key_state] = uuid4().hex
            st.success("Документ добавлен")
            st.rerun()
        except Exception as exc:
            st.error(f"Не удалось добавить документ: {exc}")


role = st.session_state.get("role")

if role is None:
    render_role_picker()
else:
    if role == "admin":
        render_admin()
    else:
        render_user()


# streamlit run src/frontend/main.py
