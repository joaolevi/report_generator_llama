import os

def load_local_data():
    from llama_index.core import SimpleDirectoryReader

    documents = SimpleDirectoryReader("./data").load_data()
    return documents

def load_s3_data(
        bucket: str,
        key: str = None,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        ):
    from llama_index.readers.file import S3Reader

    documents = S3Reader(
        bucket=bucket,
        key=key,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    ).load_data()
    return documents