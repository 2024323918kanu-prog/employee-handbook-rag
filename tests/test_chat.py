import time
from app.services.chat_service import ChatService

chat = ChatService()

question = input("Ask a question: ")

start = time.time()

response = chat.ask(question)

end = time.time()

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)
print(response["answer"])

print("\n" + "=" * 60)
print("SOURCES")
print("=" * 60)

pages = sorted(set(source["page"] for source in response["sources"]))

for page in pages:
    print(f"Employee Handbook - Page {page}")

print(f"\nResponse Time: {end - start:.2f} seconds")