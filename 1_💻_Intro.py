import logging
from logging.handlers import RotatingFileHandler
from os import mkdir
from os.path import exists

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


if __name__ == "__main__":

    # создание папки для хранения логов
    if not exists("logs"):
        mkdir("logs")

    # конфигурация параметров логирования
    logging.basicConfig(
        format="[%(asctime)s.%(msecs)03d] %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                filename="logs/streamlit.log",
                mode="a",
                maxBytes=500000,
                backupCount=5,
                delay=True,
            )
        ],
        level=logging.NOTSET,
    )

    main()
