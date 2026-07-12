import time
import requests


class OllamaClient:

    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "llama3.2:3b"

    def generate(self, prompt):

       
        raise Exception("OLLAMA CLIENT IS RUNNING")

        start = time.perf_counter()

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 150
                }
            }
        )

        print(f"Ollama responded in {time.perf_counter()-start:.2f} sec")

        return response.json()["response"]