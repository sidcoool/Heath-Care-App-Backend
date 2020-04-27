import pymongo

client = pymongo.MongoClient('mongodb+srv://Sid:Sid_smart1@cluster0-mdc4k.mongodb.net/test?retryWrites=true&w=majority')

db = client.minor
col = db.test

post = {'a': 'abc',
        'b': 'bcd' }

# postId = col.insert_one(post).inserted_id
print(col.insert_one(post))