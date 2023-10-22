import requests
import utils
import json

def get_gender(full_name):
    full_name = full_name.lower()
    first_name = utils.extract_first_name(full_name)
    if first_name is None:
        return "unknown"
    gender_payload = {"name": first_name}
    session = requests.Session()
    gender_return = session.get("https://api.genderize.io/?", params=gender_payload)
    cache_obj = json.loads(gender_return.text)
    gender = cache_obj["gender"]
    return gender

if __name__ == "__main__":
    # Test gender prediction on a list of names (requires connection to MongoDB)
    names = "Lea Frerrmann"
    result = get_gender(names)
    print(result)