import boto3
import vecs
import json

client = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id='<replace_your_own_credentials>',
    aws_secret_access_key='<replace_your_own_credentials>',
    aws_session_token='<replace_your_own_credentials>',
)
DB_CONNECTION = "postgresql://<user>:<password>@<host>:<port>/<db_name>"

query_sentence = "A quick animal jumps over a lazy one."

# create vector store client
vx = vecs.Client(DB_CONNECTION)

# create an embedding for the query sentence
response = client.invoke_model(
        body= json.dumps({"inputText": query_sentence}),
        modelId= "amazon.titan-embed-text-v1",
        accept = "application/json",
        contentType = "application/json"
    )
response_body = json.loads(response["body"].read())

query_embedding = response_body.get("embedding")

# query the 'bedrock_sentences' collection for the most similar sentences
results = bedrock_sentences.query(
    data=query_embedding,
    limit=3,
    include_value = True
)

# print the results
for result in results:
    print(result)