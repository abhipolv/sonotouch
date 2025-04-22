import argparse
import socket
from threading import Event, Thread

TCP_PORT = 3333

class TcpServer(object):
    def __init__(self, port, family_addr, persist=False, timeout=60, callback=None):
        self.port = port
        self.socket = socket.socket(family_addr, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(timeout)
        self.shutdown = Event()
        self.persist = persist
        self.family_addr = family_addr
        self.server_thread = None
        self.callback = callback

    def __enter__(self):
        try:
            self.socket.bind(('192.168.50.155', self.port))
        except socket.error as e:
            print('Bind failed:{}'.format(e))
            raise
        self.socket.listen(1)

        print(f'Starting server on port={self.port} family_addr={self.family_addr}')
        self.server_thread = Thread(target=self.run_server)
        self.server_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        if self.persist:
            sock = socket.socket(self.family_addr, socket.SOCK_STREAM)
            sock.connect(('localhost', self.port))
            sock.sendall(b'Stop', )
            sock.close()
            self.shutdown.set()
        self.shutdown.set()
        self.server_thread.join()
        self.socket.close()

    def run_server(self) -> None:
        while not self.shutdown.is_set():
            try:
                conn, address = self.socket.accept()  # accept new connection
                print(f'Connection from: {address}')
                conn.setblocking(1)
                data = conn.recv(16000) # reading 8000 samples, which is 16000 bytes
                if data and self.callback:
                    self.callback(data)
                reply = f'OK: received {len(data)} bytes, first sample is {(data[0]<< 8) | data[1]}'
                conn.send(reply.encode())
                conn.close()
            except socket.timeout:
                print(f'socket accept timeout ({self.socket.timeout}s)')
            except socket.error as e:
                print(f'Running server failed:{e}')
                raise
            if not self.persist:
                break

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=TCP_PORT, type=int, help='TCP server port')
    parser.add_argument('--ipv6', action='store_true', help='Create IPv6 server.')
    parser.add_argument('--timeout', default=10, type=int, help='socket accept/recv timeout.')
    args = parser.parse_args()

    if args.ipv6:
        family = socket.AF_INET6
    else:
        family = socket.AF_INET

    with TcpServer(args.port, family, persist=True, timeout=args.timeout):
        input('Server Running. Press Enter or CTRL-C to exit...\n')

if __name__ == '__main__':
    main()
