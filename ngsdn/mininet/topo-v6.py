#!/usr/bin/env python3
# coding=utf-8

import argparse
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.net import Mininet
from mininet.node import Host, RemoteController, Node
from mininet.topo import Topo
from mininet.link import TCLink
from stratum import StratumBmv2Switch

host_leaf_map = {
    1: ['h1a', 'h1b'],
    2: ['h2a', 'h2b'],
    3: ['h3a', 'h3b'],
    4: ['h4a', 'h4b'],
    5: ['h5a', 'h5b', 'h5c'],
}


def enable_ip_forward(node):
    node.cmd('sysctl -w net.ipv4.ip_forward=1')


class V6FabricTopo(Topo):
    def build(self):
        s1 = self.addSwitch('s1', cls=StratumBmv2Switch, grpcAddr='0.0.0.0:50006', deviceId=6, bmv2JsonFile='build/basic.json')
        s2 = self.addSwitch('s2', cls=StratumBmv2Switch, grpcAddr='0.0.0.0:50007', deviceId=7, bmv2JsonFile='build/basic.json')
        s3 = self.addSwitch('s3', cls=StratumBmv2Switch, grpcAddr='0.0.0.0:50008', deviceId=8, bmv2JsonFile='build/basic.json')

        for i in range(1, 6):
            r = self.addHost(f'r{i}', cls=Node)
            ls = self.addSwitch(
                f'ls{i}',
                cls=StratumBmv2Switch,
                grpcAddr=f'0.0.0.0:{50000 + i}',  # 50001..50005
                deviceId=i,                      # 1..5
                bmv2JsonFile='build/basic.json'
            )

            if i in (1, 2):
                self.addLink(r, s1)
            elif i == 3:
                self.addLink(r, s2)
            else:
                self.addLink(r, s3)

            self.addLink(r, ls)

            for hname in host_leaf_map[i]:
                h = self.addHost(hname)
                self.addLink(h, ls)

        r_edge = self.addHost('r_edge', cls=Node)
        ext_sw = self.addSwitch('s99_ext',
            cls=StratumBmv2Switch,
            grpcAddr='0.0.0.0:50099',
            deviceId=99,
            bmv2JsonFile='build/basic.json'
        )
        self.addLink(r_edge, ext_sw)

        # 增加三条链路，确保 eth1-eth3 存在
        self.addLink(r_edge, s1)
        self.addLink(r_edge, s2)
        self.addLink(r_edge, s3)

        for name in ('h6', 'h7', 'h8'):
            h = self.addHost(name)
            self.addLink(h, ext_sw)


def main():
    parser = argparse.ArgumentParser(description='Leaf-Spine-Edge Topo (Stratum-Bmv2)')
    parser.add_argument('--controller-ip', default='127.0.0.1', help='ONOS 控制器的 IP (默认为 127.0.0.1)')
    args = parser.parse_args()

    topo = V6FabricTopo()
    net = Mininet(topo=topo, controller=None, link=TCLink, switch=StratumBmv2Switch)

    info('*** 添加 RemoteController %s:6633\n' % args.controller_ip)
    net.addController('c0', controller=RemoteController, ip=args.controller_ip, port=6633)

    net.start()

    external_hosts = [('h6', '10.1.0.6/24'), ('h7', '10.1.0.7/24'), ('h8', '10.1.0.8/24')]
    for name, ip in external_hosts:
        h = net.get(name)
        ip_addr, mask = ip.split('/')
        h.setIP(ip_addr, prefixLen=int(mask), intf=f'{name}-eth0')
        h.cmd('ip route add default via 10.1.0.1')

    r_edge = net.get('r_edge')
    r_edge.setIP('10.1.0.1', 24, 'r_edge-eth0')
    r_edge.setIP('192.168.10.254', 24, 'r_edge-eth1')
    r_edge.setIP('192.168.20.254', 24, 'r_edge-eth2')
    r_edge.setIP('192.168.30.254', 24, 'r_edge-eth3')

    for i in range(1, 6):
        r = net.get(f'r{i}')
        if i in (1, 2):
            spine_ip = f'192.168.10.{i}'
        elif i == 3:
            spine_ip = '192.168.20.3'
        else:
            spine_ip = f'192.168.30.{i}'

        r.setIP(spine_ip, 24, f'r{i}-eth0')
        r.setIP(f'10.1.{i}.254', 24, f'r{i}-eth1')

    for i in range(1, 6):
        for idx, hname in enumerate(host_leaf_map[i], start=1):
            h = net.get(hname)
            ip_addr = f'10.1.{i}.{idx}'
            h.setIP(ip_addr, 24, f'{hname}-eth0')
            h.cmd(f'ip route add default via 10.1.{i}.254')

    enable_ip_forward(r_edge)
    for i in range(1, 6):
        r = net.get(f'r{i}')
        enable_ip_forward(r)

        if i in (1, 2):
            nh = f'192.168.10.{i}'
            gw = '192.168.10.254'
        elif i == 3:
            nh = '192.168.20.3'
            gw = '192.168.20.254'
        else:
            nh = f'192.168.30.{i}'
            gw = '192.168.30.254'

        r_edge.cmd(f'ip route add 10.1.{i}.0/24 via {nh}')
        r.cmd(f'ip route add default via {gw}')
        r.cmd(f'ip route add 10.1.0.0/24 via {gw}')

    info('*** 网络配置完成，进入 CLI\n')
    CLI(net)
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    main()
