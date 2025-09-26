#!/bin/bash


ETH=$1

# 检查是否传入了参数
if [ -z "$ETH" ]; then
    echo "Usage: $0 <network_interface>"
    exit 1
fi

# 检查out目录是否存在
if [ -d out ]; then
    echo "Directory 'out' already exists."
else
    mkdir out
    echo "Directory 'out' created."
fi

# 启动tcpdump并将数据写入out目录
tcpdump -i $ETH -w out/$ETH.pcap &
