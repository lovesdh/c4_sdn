#!/usr/bin/pyhton

from scapy.all import sniff, IPv6

def packet_callback(packet):
    if packet.haslayer(IPv6):
        ipv6_layer = packet.getlayer(IPv6)
        print("New packet: {} -> {} time:{}".format(ipv6_layer.src, ipv6_layer.dst,time.time()))

def start_sniffing(iface):
    print("Starting packet sniffer on interface {}".format(iface))
    sniff(iface=iface, prn=packet_callback, store=0)

if __name__ == "__main__":
    import sys
    iface = sys.argv[1]  
    start_sniffing(iface)




