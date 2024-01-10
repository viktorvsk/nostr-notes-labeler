import asyncio
import signal
import websockets
from worker import Worker

worker = Worker()


async def client():
    uri = "wss://saltivka.org"
    async with websockets.connect(uri) as websocket:
        await worker.on_connect(websocket)

        # Close the connection when receiving SIGTERM.
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(
            signal.SIGTERM, loop.create_task, websocket.close())

        async for message in websocket:
            await worker.on_message(websocket, message)

if __name__ == "__main__":
    asyncio.run(client())
