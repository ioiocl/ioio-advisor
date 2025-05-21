import httpx
import asyncio
import unicodedata

def normalize_text(text):
    """Normalize text by removing accents and converting to lowercase."""
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII').lower()

async def test():
    queries = [
        "Como ha evolucionado la inflacion este año?",
        "Cuales son las mejores opciones de inversion ahora?"
    ]
    
    async with httpx.AsyncClient(base_url='http://localhost:8000') as client:
        for query in queries:
            response = await client.post(
                '/query',
                json={'query': query}
            )
            data = response.json()
            print(f"\nQuery: {query}")
            print(f"Raw text: {data['text']}")
            print(f"Normalized text: {normalize_text(data['text'])}")

if __name__ == '__main__':
    asyncio.run(test())
