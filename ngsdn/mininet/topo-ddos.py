#!/usr/bin/python


import argparse

from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import Host
from mininet.topo import Topo
from stratum import StratumBmv2Switch
from mininet.node import RemoteController

CPU_PORT = 255


class IPv6Host(Host):
    """Host that can be configured with an IPv6 gateway (default route).
    """

    def config(self, ipv6, ipv6_gw=None, **params):
        super(IPv6Host, self).config(**params)
        self.cmd('ip -4 addr flush dev %s' % self.defaultIntf())
        # self.cmd('ip -4 addr add %s dev %s' % (ipv4,self.defaultIntf()))
        
        # def updateIPv4():
        #     return ipv4.split('/')[0]
        # self.defaultIntf().updateIP = updateIPv4
        
        self.cmd('ip -6 addr flush dev %s' % self.defaultIntf())
        self.cmd('ip -6 addr add %s dev %s' % (ipv6, self.defaultIntf()))
        if ipv6_gw:
            self.cmd('ip -6 route add default via %s' % ipv6_gw)
        # if ip_gw:
        #     self.cmd('ip -4 route add default via %s' % ip_gw)
        # Disable offload
        for attr in ["rx", "tx", "sg"]:
            cmd = "/sbin/ethtool --offload %s %s off" % (self.defaultIntf(), attr)
            self.cmd(cmd)

        def updateIPv6():
            return ipv6.split('/')[0]

        self.defaultIntf().updateIP = updateIPv6

    def terminate(self):
        super(IPv6Host, self).terminate()


class TutorialTopo(Topo):
    """2x2 fabric topology with IPv6 hosts"""

    def __init__(self, *args, **kwargs):
        Topo.__init__(self, *args, **kwargs)

        # Spines
        # gRPC port 50005
        spine1 = self.addSwitch('spine1', cls=StratumBmv2Switch, cpuport=CPU_PORT)
        # gRPC port 50006
        spine2 = self.addSwitch('spine2', cls=StratumBmv2Switch, cpuport=CPU_PORT)
        # gRPC port 50007
        spine3 = self.addSwitch('spine3', cls=StratumBmv2Switch, cpuport=CPU_PORT)

        #leafs
        # gRPC port 50001
        leaf1 = self.addSwitch('leaf1', cls=StratumBmv2Switch, cpuport=CPU_PORT)
        # gRPC port 50002
        leaf2 = self.addSwitch('leaf2',cls=StratumBmv2Switch,cpuport=CPU_PORT)
        # gRPC port 50003
        leaf3 = self.addSwitch('leaf3',cls=StratumBmv2Switch,cpuport=CPU_PORT)
        # gRPC port 50004
        leaf4 = self.addSwitch('leaf4',cls=StratumBmv2Switch,cpuport=CPU_PORT)
        
        # nat1 = self.addNode( 'nat1', cls=NAT, ip=natIP,inNamespace=False )
        # Switch Links
        self.addLink(spine1,spine2) #spine1 port 1 ---- spine2 port 1
        # self.addLink(spine1,leaf1) #spine1 port 2 ---- leaf1 port 1
        # self.addLink(spine2,spine3) # spine2 port 2 ---- spine3 port 1
        self.addLink(spine2,leaf1) #spine2 port 2 ---- leaf1 port 2
        self.addLink(spine2,leaf2) # spine2 port 3 ---- leaf2 port 1
        self.addLink(spine3,leaf1) # leaf1 port 2 -- spine3 port 1 
        self.addLink(spine3,leaf2) #spine3 port 2 ---- leaf2 port 2
        # self.addLink(leaf1,) # leaf1 port 3 ---- leaf2 port 3
        # self.addLink(leaf2,leaf3) # leaf2 port 4 ---- leaf3 port 1
        self.addLink(spine2,leaf3) # spine2 port 4 ---- leaf3 port 1
        self.addLink(spine3,leaf3) # spine3 port 3 ---- leaf3 port 2
        self.addLink(spine3,leaf4) # spine3 port 4 ---- leaf4 port 1
        self.addLink(spine2,leaf4) # spine2 port 5 ---- leaf4 port 2
        # self.addLink(leaf3,leaf4) # leaf3 port 4 ---- leaf4 port 2
        

        # IPv6 hosts attached to leaf 1
        h1 = self.addHost('h1', cls=IPv6Host, mac="00:00:00:00:00:10",
                        
                           ipv6='2001:1:1::1/64', ipv6_gw='2001:1:1::ff')
        # h1b = self.addHost('h1b', cls=IPv6Host, mac="00:00:00:00:00:1B",
        #                
        #                    ipv6='2001:1:1::b/64', ipv6_gw='2001:1:1::ff')
        # h1c = self.addHost('h1c', cls=IPv6Host, mac="00:00:00:00:00:1C",
        #               
        #                    ipv6='2001:1:1::c/64', ipv6_gw='2001:1:1::ff')
        h2 = self.addHost('h2', cls=IPv6Host, mac="00:00:00:00:00:20",
                          ipv6='2001:1:2::1/64', ipv6_gw='2001:1:2::ff')
        
        h3a = self.addHost('h3a',cls=IPv6Host,mac="00:00:00:00:00:3A",
                          ipv6='2001:1:3::a/64',ipv6_gw='2001:1:3::ff')
        h3b = self.addHost('h3b',cls=IPv6Host,mac="00:00:00:00:00:3B",
                          ipv6='2001:1:3::b/64',ipv6_gw='2001:1:3::ff')
        h3c = self.addHost('h3c',cls=IPv6Host,mac="00:00:00:00:00:3C",
                          ipv6='2001:1:3::c/64',ipv6_gw='2001:1:3::ff')
        
        h4a = self.addHost('h4a',cls=IPv6Host,mac="00:00:00:00:00:4A",
                          ipv6='2001:1:4::a/64',ipv6_gw='2001:1:4::ff')
        h4b = self.addHost('h4b',cls=IPv6Host,mac="00:00:00:00:00:4B",
                          ipv6='2001:1:4::b/64',ipv6_gw='2001:1:4::ff')
        h4c = self.addHost('h4c',cls=IPv6Host,mac="00:00:00:00:00:4C",
                          ipv6='2001:1:4::c/64',ipv6_gw='2001:1:4::ff')
        
        self.addLink(h1,leaf1) # port 3
        self.addLink(h2,leaf2) # port 3
        self.addLink(h3a,leaf3) # port 3
        self.addLink(h3b,leaf3) #port 4
        self.addLink(h3c,leaf3) # port 5
        self.addLink(h4a,leaf4) # port 3
        self.addLink(h4b,leaf4) # port 4
        self.addLink(h4c,leaf4) # port 5
        

controller_ip = 'localhost'
controller_port = 6633
def main():
    net = Mininet(topo=TutorialTopo(), controller=None)
    # net.addController('controller', controller=RemoteController, ip=controller_ip, port=controller_port)
    # net.addNAT().configDefault()
    net.start()
    # h1 = net.get('h1')
    # h1.cmd('ping leaf1')
    
    # h2 = net.get('h2')
    # h2.cmd('ping leaf2')
    
    # h3a = net.get('h3a')
    # h3a.cmd('ping leaf3')
    # h3b = net.get('h3b')
    # h3b.cmd('ping leaf3')
    # h3c = net.get('h3c')
    # h3c.cmd('ping leaf3')
    # h4a = net.get('h4a')
    # h4a.cmd('ping leaf4')
    # h4b = net.get('h4b')
    # h4b.cmd('ping leaf4')
    # h4c = net.get('h4c')
    # h4c.cmd('ping leaf4')
    
    # result = h2.cmd('ifconfig')
    # print(h2.cmd('python3 /mininet/host-service/httpSimpleServer.py &'))
    # print(result)
    CLI(net)
    net.stop()
    # print '#' * 80
    # print 'ATTENTION: Mininet was stopped! Perhaps accidentally?'
    # print 'No worries, it will restart automatically in a few seconds...'
    # print 'To access again the Mininet CLI, use `make mn-cli`'
    # print 'To detach from the CLI (without stopping), press Ctrl-D'
    # print 'To permanently quit Mininet, use `make stop`'
    # print '#' * 80


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Mininet topology script for 2x2 fabric with stratum_bmv2 and IPv6 hosts')
    args = parser.parse_args()
    setLogLevel('info')

    main()
