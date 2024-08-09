import hashlib
import secrets
import os

def generate_salt(length=16): return secrets.token_bytes(length)

def hash_exam_id(exam_id, salt=None):
    if salt is None: salt = generate_salt() 
    hash_input = salt + exam_id.encode('utf-8') 
    hash_object = hashlib.sha256(hash_input)
    hashed_id = hash_object.hexdigest()
    return hashed_id, salt

# Example usage
exam_id = '12345'
hashed_id, used_salt = hash_exam_id(exam_id)
print(f"Hashed ID: {hashed_id}")
print(f"Used Salt (hex): {used_salt.hex()}") 