import boto3
import vecs
import json

client = boto3.client(
    'bedrock-runtime',
    region_name='us-west-2',
    aws_access_key_id='{replace_your_own_credentials}',
    aws_secret_access_key='{replace_your_own_credentials}',
    aws_session_token='{replace_your_own_credentials}',
)
DB_CONNECTION = "postgresql://postgres.<user>:<password>@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"


dataset = [
    "The cat sat on the mat.",
    "The quick brown fox jumps over the lazy dog.",
    "Friends, Romans, countrymen, lend me your ears",
    "To be or not to be, that is the question.",
]

embeddings = []

for sentence in dataset:
    response = client.invoke_model(
        body= json.dumps({"inputText": sentence}),
        modelId= "amazon.titan-embed-text-v1",
        accept = "application/json",
        contentType = "application/json"
    )
    response_body = json.loads(response["body"].read())
    embeddings.append((sentence, response_body.get("embedding"), {}))

# create vector store client
vx = vecs.Client(DB_CONNECTION)

# create a collection named 'bedrock_sentences' with 1536 dimensional vectors (default dimension for text-embedding-ada-002)
bedrock_sentences = vx.get_or_create_collection(name="bedrock_sentences", dimension=1536)

# upsert the embeddings into the 'sentences' collection
bedrock_sentences.upsert(records=embeddings)

# create an index for the 'sentences' collection
bedrock_sentences.create_index()