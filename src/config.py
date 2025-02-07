import os
from dotenv import load_dotenv
import boto3
from langchain_community.embeddings import BedrockEmbeddings

def load_config():
    # Load environment variables from .env file
    load_dotenv()

    global s3_client, BUCKET_NAME, bedrock_client, bedrock_embeddings, folder_path

    s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    if not BUCKET_NAME:
        raise ValueError("BUCKET_NAME environment variable is not set")

    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1"
    )
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock_client)

    folder_path = "/tmp/"

# Expose the variables for import
load_config()
s3_client = s3_client
BUCKET_NAME = BUCKET_NAME
bedrock_client = bedrock_client
bedrock_embeddings = bedrock_embeddings
folder_path = folder_path