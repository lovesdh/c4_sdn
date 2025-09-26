#!/bin/pyhotn

import socket
import sys

from scapy.all import sendp, Ether, IPv6, UDP

def send_packet(dst_mac, src_ip, dst_ip, iface):
    packet = Ether(dst=dst_mac) / IPv6(src=src_ip, dst=dst_ip) / UDP()
    sendp(packet, iface=iface)
    print("Packet sent from {} to {} via interface {}".format(src_ip, dst_ip, iface))

if __name__ == "__main__":
    
    dst_mac = "00:00:00:00:00:20"  
    src_ip = "2001:1:1::1"
    dst_ip = "2001:1:2::1"
    iface = "h1-eth0" 
    while True: 
        send_packet(dst_mac, src_ip, dst_ip, iface)
        time.sleep(0.5)


