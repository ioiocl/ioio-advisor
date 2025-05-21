import httpx
import asyncio

async def test():
    async with httpx.AsyncClient(base_url='http://localhost:8000') as client:
        response = await client.post(
            '/query',
            json={'query': 'Como ha evolucionado la inflacion este año?'}
        )
        print(response.json())

if __name__ == '__main__':
    asyncio.run(test())
