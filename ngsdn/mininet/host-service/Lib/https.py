import http.server
import socket
import socketserver

class IPv6HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Customize the log message format
        self.log_date_time_string = lambda: ''  # Remove date and time
        http.server.SimpleHTTPRequestHandler.log_message(self, format, *args)

def run(server_class=http.server.HTTPServer, handler_class=IPv6HTTPRequestHandler, port=8000, ipv6_address="2001:1:1::1"):
    server_address = (ipv6_address, port)
    httpd = server_class(server_address, handler_class)
    print("Starting httpd server on 2001:1:1::1:8080")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
