from openai import OpenAI
import vecs

client = OpenAI(api_key='replace_your_own_credentials')
DB_CONNECTION = "postgresql://postgres.<user>:<password>@<host>:<port>/<db_name>"

query_sentence = "A quick animal jumps over a lazy one."

# create vector store client
vx = vecs.Client(DB_CONNECTION)

# create an embedding for the query sentence
response = client.embeddings.create(model="text-embedding-ada-002",
    input=[query_sentence])


# response = openai.Embedding.create(
#     model="text-embedding-ada-002",
#     input=[query_sentence]
# )
query_embedding = response.data[0].embedding
print(query_embedding)

sentences = vx.get_or_create_collection(name="sentences", dimension=1536)

# query the 'sentences' collection for the most similar sentences
results = sentences.query(
    data=query_embedding,
    limit=3,
    include_value = True
)

# print the results
for result in results:
    print(result)