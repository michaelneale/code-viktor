import argparse
from sentence_transformers import SentenceTransformer, util
import torch

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def load_vector_database(db_path):
    data = torch.load(db_path)
    return data['file_paths'], data['embeddings']


def find_similar_files(query, file_paths, embeddings):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=5)
    similar_files = [file_paths[idx] for idx in top_results[1]]
    return similar_files


def main():
    parser = argparse.ArgumentParser(description='Query a vector database to find similar files.')
    parser.add_argument('query', type=str, help='Query string to find similar files')
    args = parser.parse_args()

    db_path = '.code-vectors.pt'
    file_paths, embeddings = load_vector_database(db_path)
    similar_files = find_similar_files(args.query, file_paths, embeddings)
    for file in similar_files:
        print(file)


if __name__ == '__main__':
    main()
