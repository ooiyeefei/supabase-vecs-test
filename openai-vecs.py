from openai import OpenAI
import vecs

client = OpenAI(api_key='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')
DB_CONNECTION = "postgresql://postgres.oenhkohxyjgrspjcpiuh:qZBvrX3lx2Aplnma@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"


dataset = [
    "The cat sat on the mat.",
    "The quick brown fox jumps over the lazy dog.",
    "Friends, Romans, countrymen, lend me your ears",
    "To be or not to be, that is the question.",
]

embeddings = []

for sentence in dataset:
    response = client.embeddings.create(model="text-embedding-ada-002",
    input=[sentence])
    embeddings.append((sentence, response.data[0].embedding, {}))

# create vector store client
vx = vecs.Client(DB_CONNECTION)

# create a collection named 'sentences' with 1536 dimensional vectors (default dimension for text-embedding-ada-002)
sentences = vx.get_or_create_collection(name="sentences", dimension=1536)

# upsert the embeddings into the 'sentences' collection
sentences.upsert(records=embeddings)

# create an index for the 'sentences' collection
sentences.create_index()