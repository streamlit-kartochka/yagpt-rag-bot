# создаем простое streamlit приложение для работы с вашими pdf-файлами при помощи YaGPT

import logging
import os
import tempfile

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch
from streamlit_chat import message
from tenacity import retry, stop_after_attempt, wait_fixed
from yandex_chain import YandexEmbeddings, YandexLLM

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIRECTORY = '.'

# использовать системные переменные из облака streamlit (secrets)
yagpt_api_key = st.secrets['yagpt_api_key']
yagpt_folder_id = st.secrets['yagpt_folder_id']
mdb_os_pwd = st.secrets['mdb_os_pwd']
mdb_os_hosts = st.secrets['mdb_os_hosts'].split(',')
mdb_os_index_name = st.secrets['mdb_os_index_name']
MDB_OS_CA = st.secrets['mdb_os_ca']


# Функция для повторных попыток
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def create_embeddings(folder_id, api_key):
    try:
        embeddings = YandexEmbeddings(folder_id=folder_id, api_key=api_key)
        # Проверка работоспособности эмбеддингов
        test_embedding = embeddings.embed_query('test')
        logger.info(
            f'Embeddings initialized successfully. Test embedding shape: {len(test_embedding)}'
        )
        return embeddings
    except Exception as e:
        logger.error(f'Error initializing embeddings: {str(e)}')
        raise


def check_opensearch_connection(hosts, auth, use_ssl, verify_certs, ca_certs):
    try:
        client = OpenSearch(
            hosts=hosts,
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ca_certs=ca_certs,
        )
        # Проверка подключения
        info = client.info()
        logger.info(
            f"Successfully connected to OpenSearch. Cluster name: {info['cluster_name']}"
        )
        return client
    except Exception as e:
        logger.error(f'Error connecting to OpenSearch: {str(e)}')
        raise


def ingest_docs(temp_dir: str = tempfile.gettempdir()):
    """
    Инъекция ваших pdf файлов в MBD Opensearch
    """
    try:
        # выдать ошибку, если каких-то переменных не хватает
        if (
            not yagpt_api_key
            or not yagpt_folder_id
            or not mdb_os_pwd
            or not mdb_os_hosts
            or not mdb_os_index_name
        ):
            raise ValueError(
                'Пожалуйста укажите необходимый набор переменных окружения'
            )

        # загрузить PDF файлы из временной директории
        loader = DirectoryLoader(
            temp_dir,
            glob='**/*.pdf',
            loader_cls=PyPDFLoader,
            recursive=True,
        )
        documents = loader.load()

        # разбиваем документы на блоки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        documents = text_splitter.split_documents(documents)
        print(len(documents))
        text_to_print = f'Ориентировочное время = {len(documents)} с.'
        st.text(text_to_print)

        # подключаемся к базе данных MDB Opensearch, используя наши ключи (проверка подключения)
        opensearch_client = check_opensearch_connection(
            mdb_os_hosts,
            ('admin', mdb_os_pwd),
            use_ssl=True,
            verify_certs=True,
            ca_certs=MDB_OS_CA,
        )

        # инициируем процедуру превращения блоков текста в Embeddings через YaGPT Embeddings API, используя API ключ доступа
        embeddings = create_embeddings(yagpt_folder_id, yagpt_api_key)

        # добавляем "документы" (embeddings) в векторную базу данных Opensearch
        try:
            docsearch = OpenSearchVectorSearch.from_documents(
                documents,
                embeddings,
                opensearch_url=mdb_os_hosts,
                http_auth=('admin', mdb_os_pwd),
                use_ssl=True,
                verify_certs=True,
                ca_certs=MDB_OS_CA,
                engine='lucene',
                index_name=mdb_os_index_name,
                bulk_size=1000000,
                client=opensearch_client,  # Используем проверенный клиент
            )
            logger.info(
                f'Successfully created OpenSearchVectorSearch index: {mdb_os_index_name}'
            )
        except Exception as e:
            logger.error(f'Error creating OpenSearchVectorSearch: {str(e)}')
            raise

    except Exception as e:
        logger.error(f'Возникла ошибка при добавлении ваших файлов: {str(e)}')
        raise


# это основная функция, которая запускает приложение streamlit
def main():
    # Загрузка логотипа компании
    logo_image = './images/logo.png'  # Путь к изображению логотипа

    # Отображение логотипа в основной части приложения
    from PIL import Image

    # Загрузка логотипа
    logo = Image.open(logo_image)
    # Изменение размера логотипа
    resized_logo = logo.resize((100, 100))
    # Отображаем лого измененного небольшого размера
    st.image(resized_logo)
    # Указываем название и заголовок Streamlit приложения
    st.title('YaGPT-чат с вашими PDF файлами')
    st.warning(
        'Загружайте свои PDF-файлы и задавайте вопросы по ним. Если вы уже загрузили свои файлы, то ***обязательно*** удалите их из списка загруженных и переходите к чату ниже.'
    )

    # вводить все credentials в графическом интерфейсе слева
    # Sidebar contents
    with st.sidebar:
        st.title('\U0001f917\U0001f4acИИ-помощник')
        st.markdown("""
        ## О программе
        Данный YaGPT-помощник реализует [Retrieval-Augmented Generation (RAG)](https://github.com/yandex-cloud-examples/yc-yandexgpt-qa-bot-for-docs/blob/main/README.md) подход
        и использует следующие компоненты:
        - [Yandex GPT](https://cloud.yandex.ru/services/yandexgpt)
        - [Yandex GPT for Langchain](https://pypi.org/project/yandex-chain/)
        - [YC MDB Opensearch](https://cloud.yandex.ru/docs/managed-opensearch/)
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        """)

    global \
        yagpt_folder_id, \
        yagpt_api_key, \
        mdb_os_ca, \
        mdb_os_pwd, \
        mdb_os_hosts, \
        mdb_os_index_name

    mdb_os_ca = MDB_OS_CA

    yagpt_temp = st.sidebar.text_input(
        'Температура', type='password', value=0.01
    )
    rag_k = st.sidebar.text_input(
        'Количество поисковых выдач размером с один блок',
        type='password',
        value=5,
    )

    # Параметры chunk_size и chunk_overlap
    global chunk_size, chunk_overlap
    chunk_size = st.sidebar.slider(
        "Выберите размер текстового 'окна' разметки документов в символах",
        0,
        2000,
        1000,
    )
    chunk_overlap = st.sidebar.slider(
        'Выберите размер блока перекрытия в символах', 0, 400, 100
    )

    # Выводим предупреждение, если пользователь не указал свои учетные данные
    if (
        not yagpt_api_key
        or not yagpt_folder_id
        or not mdb_os_pwd
        or not mdb_os_hosts
        or not mdb_os_index_name
    ):
        st.warning(
            'Пожалуйста, задайте свои учетные данные (в secrets/.env или в раскрывающейся панели слева) для запуска этого приложения.'
        )

    # Загрузка pdf файлов
    uploaded_files = st.file_uploader(
        'После загрузки файлов в формате pdf начнется их добавление в векторную базу данных MDB Opensearch.',
        accept_multiple_files=True,
        type=['pdf'],
    )

    # если файлы загружены, сохраняем их во временную папку и потом заносим в vectorstore
    if uploaded_files:
        # создаем временную папку и сохраняем в ней загруженные файлы
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    # сохраняем файл во временную папку
                    with open(os.path.join(temp_dir, file_name), 'wb') as f:
                        f.write(uploaded_file.read())
                # отображение спиннера во время инъекции файлов
                with st.spinner('Добавление ваших файлов в базу ...'):
                    ingest_docs(temp_dir)
                    st.success('Ваш(и) файл(ы) успешно принят(ы)')
                    st.session_state['ready'] = True
        except Exception as e:
            st.error(f'При загрузке ваших файлов произошла ошибка: {str(e)}')

    # Логика обработки сообщений от пользователей
    # инициализировать историю чата, если ее пока нет
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # инициализировать состояние готовности, если его пока нет
    if 'ready' not in st.session_state:
        st.session_state['ready'] = True

    if st.session_state['ready']:
        # подключиться к векторной БД Opensearch, используя учетные данные (проверка подключения)
        conn = check_opensearch_connection(
            mdb_os_hosts,
            ('admin', mdb_os_pwd),
            use_ssl=True,
            verify_certs=True,
            ca_certs=MDB_OS_CA,
        )

        # инициализировать модели YandexEmbeddings и YandexGPT
        embeddings = create_embeddings(yagpt_folder_id, yagpt_api_key)

        # обращение к модели YaGPT
        llm = YandexLLM(
            api_key=yagpt_api_key,
            folder_id=yagpt_folder_id,
            temperature=yagpt_temp,
            max_tokens=7000,
        )

        # инициализация retrival chain - цепочки поиска
        vectorstore = OpenSearchVectorSearch(
            embedding_function=embeddings,
            index_name=mdb_os_index_name,
            opensearch_url=mdb_os_hosts,
            http_auth=('admin', mdb_os_pwd),
            use_ssl=True,
            verify_certs=True,
            ca_certs=MDB_OS_CA,
            engine='lucene',
        )

        template = """Представь, что ты полезный ИИ-помощник. Твоя задача отвечать на вопросы на русском языке в рамках предоставленного ниже текста.
        Отвечай точно в рамках предоставленного текста, даже если тебя просят придумать.
        Отвечай вежливо в официальном стиле. Eсли знаешь больше, чем указано в тексте, а внутри текста ответа нет, отвечай 
        "Я могу давать ответы только по тематике загруженных документов. Мне не удалось найти в документах ответ на ваш вопрос."
        {context}
        Вопрос: {question}
        Твой ответ:
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
        qa = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': rag_k}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': QA_CHAIN_PROMPT},
        )

        if 'generated' not in st.session_state:
            st.session_state['generated'] = [
                'Что бы вы хотели узнать о документе?'
            ]

        if 'past' not in st.session_state:
            st.session_state['past'] = ['Привет!']

        # контейнер для истории чата
        response_container = st.container()

        # контейнер для текстового поля
        container = st.container()
        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input(
                    'Вопрос:', placeholder='О чем этот документ?', key='input'
                )
                submit_button = st.form_submit_button(label='Отправить')

            if submit_button and user_input:
                # отобразить загрузочный "волчок"
                with st.spinner('Думаю...'):
                    try:
                        logger.info(
                            f'Обработка запроса пользователя: {user_input}'
                        )
                        output = qa({'query': user_input})
                        logger.info('Ответ успешно сгенерирован')
                        st.session_state['past'].append(user_input)
                        st.session_state['generated'].append(output['result'])

                        # обновляем историю чата с помощью вопроса пользователя и ответа от бота
                        st.session_state['chat_history'].append(
                            {'вопрос': user_input, 'ответ': output['result']}
                        )
                        ## добавляем источники к ответу
                        input_documents = output['source_documents']
                        i = 0
                        for doc in input_documents:
                            source = doc.metadata['source']
                            page_content = doc.page_content
                            i = i + 1
                            with st.expander(f'**Источник N{i}:** [{source}]'):
                                st.write(page_content)
                    except Exception as e:
                        logger.error(f'Ошибка при генерации ответа: {str(e)}')
                        st.error(
                            f'Произошла ошибка при обработке вашего запроса: {str(e)}'
                        )

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(
                        st.session_state['past'][i],
                        is_user=True,
                        key=str(i) + '_user',
                    )
                    message(st.session_state['generated'][i], key=str(i))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        error_message = f'Произошла ошибка при выполнении приложения: {str(e)}'
        logger.error(error_message)
        st.error(error_message)
        st.write(
            'Пожалуйста, проверьте настройки и попробуйте снова. Если проблема сохраняется, обратитесь к администратору.'
        )
