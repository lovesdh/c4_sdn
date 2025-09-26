#!/usr/bin/python

import socket

class SimpleHTTPServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None

    def start(self):
        try:
            # 创建一个TCP IPv6套接字
            self.server_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            # 设置套接字选项，允许地址重用
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # 绑定主机和端口
            self.server_socket.bind((self.host, self.port, 0, 0))
            # 监听连接
            self.server_socket.listen(5)
            print("HTTP server running on %s, port %d" % (self.host, self.port))

            while True:
                # 接受连接
                client_socket, client_address = self.server_socket.accept()
                print("Received connection from %s" % str(client_address))
                # 处理HTTP请求
                self.handle_request(client_socket)

        except Exception as e:
            print(str(e))

        finally:
            if self.server_socket:
                self.server_socket.close()
                print("HTTP server closed")

    def handle_request(self, client_socket):
        # 接收客户端请求数据
        request_data = client_socket.recv(1024).decode('utf-8')
        # 解析请求
        request_lines = request_data.split('\n')
        if request_lines:
            # 获取请求方法、路径和HTTP版本
            method, path, http_version = request_lines[0].strip().split()
            print("HTTP method %s, path: %s, HTTP version: %s" % (method, path, http_version))
            
            # 构造HTTP响应
            response_body = "<html><body><h1>Hello, World!</h1></body></html>"
            response_headers = [
                "HTTP/1.1 200 OK",
                "Content-Type: text/html",
                "Content-Length: %d" % len(response_body),
                "\n"
            ]
            response = '\n'.join(response_headers) + response_body
            
            # 发送响应数据到客户端
            client_socket.sendall(response.encode('utf-8'))
            # 关闭客户端连接
            client_socket.close()
            print("Response sent")

# 主函数
if __name__ == "__main__":
    # 服务器主机和端口
    HOST = ''  # 绑定到所有可用的IPv6地址
    PORT = 8080
    # 创建HTTP服务器实例并启动
    http_server = SimpleHTTPServer(HOST, PORT)
    http_server.start()
