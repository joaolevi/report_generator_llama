"""Package imports"""
import pytesseract
import os
import json
from transformers import AutoTokenizer
from pdf2image import convert_from_bytes
import requests

"""Llama Imports"""
from llama_index.core import Settings
from llama_index.core import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama.base import DEFAULT_REQUEST_TIMEOUT, Ollama

class Llama3():
    def __init__(self, logger: object = None):
        self.__logger = logger

        base_url = os.getenv("LLAMA_BASE_URL") or "http://127.0.0.1:11434"
        request_timeout = float(os.getenv("LLAMA_REQUEST_TIMEOUT") or DEFAULT_REQUEST_TIMEOUT)
        Settings.embed_model = OllamaEmbedding(
            base_url=base_url,
            model_name= os.getenv("EMBEDDED_MODEL") or "all-minilm",
        )
        Settings.tokenizer = AutoTokenizer.from_pretrained(
            (os.getenv('TOKENIZER_MODEL') or 'neuralmind/bert-large-portuguese-cased'), 
            do_lower_case=False,
            force_download=True
        )
        self.__llm = Settings.llm = Ollama(
            base_url=base_url, 
            model=(os.getenv("LLM_MODEL") or "llama3"), 
            request_timeout=request_timeout
        )

    def json_formater(self, response_text: str) -> json:
        start_marker = '```json'
        end_marker = '```'
        start_index = response_text.find(start_marker) + len(start_marker)
        end_index = response_text.find(end_marker, start_index)
        json_text = response_text[start_index:end_index].strip()
        result_json = json.loads(json_text)
        return result_json

    def vectorization_mode_ocr(self, data: dict) -> json:
        from llama_index.core import VectorStoreIndex

        try:
            images = convert_from_bytes(requests.get(data['url']).content)

            documents = []
            for index, image in enumerate(images):
                texto = pytesseract.image_to_string(image, lang='por')
                doc = Document(text=texto)
                documents.append(doc)

            index = VectorStoreIndex.from_documents(documents, verbose=True)

            response = index.as_query_engine().query(data['prompt'])
            return self.json_formater(str(response))
        except Exception as e:
            return str(e)
    
    def chat_mode_ocr(self, data: dict) -> str:
        from llama_index.core.llms import ChatMessage, MessageRole

        url = data['url']
        prompt = data['prompt']
        perguntas = data['perguntas']
        campos = data['campos']

        images = convert_from_bytes(requests.get(url).content)

        full_text = ""
        for index, image in enumerate(images):
            texto = pytesseract.image_to_string(image, lang='por')
            full_text += f"Texto extraído da página {index+1}: {texto}"

        messages = [
            ChatMessage(role=message['role'], content=message['content']) 
                for message in prompt
        ]

        if perguntas and campos:
            messages.extend(
                ChatMessage(role=MessageRole.USER, content="Você deve responder as perguntas abaixo com base no texto extraído do documento. Perguntas: "
                            +" ".join(perguntas)
                            +" Responda somente com um objeto JSON contendo os seguintes campos: "
                            +" ".join(campos)),
                ChatMessage(role=MessageRole.ASSISTANT, content="Certo, vou responder as perguntas e devolver em um único objeto JSON. Me informe o texto extraído do documento."),
                ChatMessage(role=MessageRole.USER, content="Texto extraído do documento: "+full_text))
        resp = self.__llm.chat(messages)
        return self.json_formater(resp)