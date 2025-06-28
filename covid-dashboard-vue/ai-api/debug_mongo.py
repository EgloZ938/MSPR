from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()

client = MongoClient(os.getenv('MONGO_URI'))
db = client[os.getenv('DB_NAME')]

print('=== DEBUG COLLECTIONS ===')
print('Collections:', db.list_collection_names())
print()

print('=== COUNTRIES ===') 
countries = list(db.countries.find({}).limit(3))
for c in countries:
    print('Country:', c)
print('Total countries:', db.countries.count_documents({}))
print()

print('=== DAILY_STATS ===')
stats = list(db.daily_stats.find({}).limit(3))
for s in stats:
    print('Stat:', s)
print('Total stats:', db.daily_stats.count_documents({}))