import os
import argparse
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def scan_repository(repo_path):
    file_contents = []
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.py', '.java', '.js', '.cpp', '.c', '.h', '.rb', '.go', '.rs', '.php', '.html', '.css', '.md', '.dart')):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                try:
                    with open(file_path, 'r', errors='ignore') as f:
                        content = f.read()
                        file_contents.append(content)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return file_paths, file_contents


def build_vector_database(file_contents):
    embeddings = model.encode(file_contents, convert_to_tensor=True)
    return embeddings


def save_vector_database(file_paths, embeddings, db_path):
    torch.save({'file_paths': file_paths, 'embeddings': embeddings}, db_path)


def main():
    parser = argparse.ArgumentParser(description='Build a vector database for a source code repository.')
    parser.add_argument('repo_path', type=str, help='Path to the source code repository')
    args = parser.parse_args()
    db_path = '.code-vectors.pt'

    file_paths, file_contents = scan_repository(args.repo_path)
    embeddings = build_vector_database(file_contents)
    save_vector_database(file_paths, embeddings, db_path)


if __name__ == '__main__':
    main()
