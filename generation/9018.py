# from socket import *
#
# IP = '27.17.52.252'
# SEVER_PORT = 9018
# BUFLEN = 1024
#
# dataSocket = socket(AF_INET, SOCK_STREAM)
#
# # 连接服务端socket
# dataSocket.connect((IP, SEVER_PORT))
# while True:
#     tosend = input('>>')
#     if tosend == 'exit':
#         break
#     # 客户端发送消息，编码为bytes
#     dataSocket.send(tosend.encode())
#     # 等待服务端发送消息
#     recved = dataSocket.recv(BUFLEN)
#     if not recved:
#         break
#     # 打印来自服务端的消息
#     # print(recved.decode())
# dataSocket.close()

#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#
import zmq

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.PAIR)
socket.connect("tcp://27.17.52.252:9018")

#  Do 10 requests, waiting each time for a response
for request in range(5):
    #  Get the reply.
    message = socket.recv()
    print(f"Received reply  {message} ")
    # print(f"Sending request {request} …")
    # socket.send(b"Hello")

