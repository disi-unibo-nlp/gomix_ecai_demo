import openai
import os
from tenacity import retry, stop_after_attempt, wait_random_exponential

openai.api_key = os.environ.get('OPENAI_API_KEY')


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def calc_embedding(text, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
