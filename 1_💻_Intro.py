import logging

import client_funcs
import streamlit as st
from settings import Settings

settings = Settings()


def main():
    logging.info("Intro opened")
    st.set_page_config(layout="wide", page_title="Intro", page_icon="💻")
    st.title(
        "Intro. Предсказание динамики физической системы с \
             помощью нейросетей (годовой проект)"
    )
    st.markdown(
        """
                Куратор - Марк Блуменау.

                Команда:
                - Михаил Мокроносов;
                - Владислав Семенов;
                - Матвей Спиридонов.
                """
    )
    st.markdown(
        "Цель проекта - провести многоклассовую \
                классификацию событий (частиц) в эксперименте \
                LHCb (детектор на адронном коллайдере). Будут \
                опробованы как classic ML, так и DL подходы."
    )
    st.markdown(
        "На странице **📊 EDA** доступен разведочный анализ данных, \
                на странице **🤖 Classic ML** -- обучение и инференс моделей."
    )
    st.markdown(
        "Репозиторий проекта доступен по " f"[ссылке]({settings.GITHUB_URL}).")

    response = client_funcs.get()
    if response.status_code == 200:
        st.info("Соединение с сервером установлено.")
    else:
        st.error(
            "Не удалось соедниниться с сервером. "
            f"Код ошибки: {response.status_code}."
        )


main()
# if __name__ == "__main__":
#    main()
