import os
import streamlit as st
from config import s3_client, BUCKET_NAME, bedrock_embeddings, folder_path
from langchain_community.vectorstores import FAISS

def load_index(style_key):
    """
    Downloads the FAISS index files from S3 for the given style_key,
    and loads them locally so we can query them.
    
    style_key should be one of ['mail', 'normal', 'report', 'feedback'].
    """
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the relevant file names based on style_key
    faiss_index_name = f"my_faiss_{style_key}"
    faiss_index_file = faiss_index_name + ".faiss"
    faiss_pkl_file = faiss_index_name + ".pkl"
    
    files_to_download = [faiss_index_file, faiss_pkl_file]
    
    try:
        # Check if the files exist in S3, then download them
        for file_name in files_to_download:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=file_name)
            s3_client.download_file(
                Bucket=BUCKET_NAME, 
                Key=file_name, 
                Filename=os.path.join(folder_path, file_name)
            )
            st.success(f"Successfully downloaded {file_name} to {folder_path}")
    except Exception as e:
        st.error(f"Error downloading files from S3 for style '{style_key}': {str(e)}")
        return None

    # Now try loading the FAISS index from local disk
    try:
        faiss_index = FAISS.load_local(
            index_name=faiss_index_name,
            folder_path=folder_path,
            embeddings=bedrock_embeddings
        )
        return faiss_index
    except Exception as e:
        st.error(f"Error loading FAISS index for style '{style_key}': {str(e)}")
        return None