from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from sentence_transformers.util import cos_sim

load_dotenv()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

documents = ["Sri Lanka is an island nation located in the Indian Ocean, just south of India",
             "Its capital city is Sri Jayawardenepura Kotte, while Colombo is the largest city and main commercial "
             "center.",
             "The country is famous for its beautiful beaches, lush tea plantations, and rich biodiversity.",
             "Sri Lanka has a long history with ancient cities, temples, and UNESCO World Heritage Sites.",
             "The nation is known for its cultural diversity, with multiple ethnic groups, languages, and religions "
             "coexisting peacefully. ",
             ]

sentence_vectors = embeddings.embed_documents(documents)
print(sentence_vectors[0])
print("length of vectors", len(sentence_vectors[0]))


def semantic_searching(search_query, top_k2=5):
    # methna krnne api dena search text ekth vectors widiyata harawa gnnw
    search_query_vector = embeddings.embed_query(search_query)  # creating query vector

    # cosine similarity -1 <= val <=+1
    # meka krnne douclumet eke thiyenwa ewayen api search query eke eva galapend blnn compaire krnn wage

    scores = []
    for i, vec in enumerate(sentence_vectors):
        score = cos_sim(search_query_vector, vec)[0][0]  # calculate cosine similarity
        scores.append((i, score))

    print(scores)
    scores.sort(key=lambda x: x[1], reverse=True)  # sort scores by descending order

    return [(documents[index], cos_score) for index, cos_score in scores[:top_k2]]



search_result =semantic_searching("What is the best place to in Sri lakanaka ? ")


for doc , score in search_result:
    print(f"Score {score}| {doc}")