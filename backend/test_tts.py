import asyncio
import edge_tts

async def main():
    try:
        print("Starting comm stream...")
        comm = edge_tts.Communicate("Hello world", "en-US-AriaNeural")
        async for chunk in comm.stream():
            print(chunk["type"])
    except Exception as e:
        print("ERROR:", type(e).__name__, e)

asyncio.run(main())
