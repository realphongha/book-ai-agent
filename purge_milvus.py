import yaml
from pymilvus import connections, utility

# Load configuration
cfg = yaml.safe_load(open('configs.yaml', 'r'))
collection_name = "books"
connection_alias = "default"

# Connect to Milvus using the new API
print(f"Connecting to Milvus at {cfg['milvus']['host']}:{cfg['milvus']['port']}...")
connections.connect(
    alias=connection_alias,
    host=cfg['milvus']['host'],
    port=cfg['milvus']['port']
)

# Check if the collection exists and drop it
if utility.has_collection(collection_name, using=connection_alias):
    print(f"Dropping collection: {collection_name}")
    utility.drop_collection(collection_name, using=connection_alias)
    print("Collection dropped successfully.")
else:
    print(f"Collection '{collection_name}' does not exist, nothing to do.")

# Disconnect from Milvus
print("Disconnecting from Milvus.")
connections.disconnect(connection_alias)

