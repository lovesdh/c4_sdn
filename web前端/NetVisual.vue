<template>
    <div id="graph"></div>
</template>

<script>
import { DataSet } from "vis-data/peer/esm/vis-data"
import { Network } from "vis-network/peer/esm/vis-network";

export default {
    data() {
        return {
            network: null,
            nodes: null,
            edges: null,
            isAttacked: false
        };
    },
    mounted() {
        this.initNetwork();
    },
    methods: {
        initNetwork() {
            // 节点数据
            this.nodes = new DataSet([
                {
                    id: 1,
                    label: "Edge Router",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 2,
                    label: "Ryu Controller",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 3,
                    label: "Spine1",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 4,
                    label: "Spine2",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 5,
                    label: "Spine3",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 6,
                    label: "Leaf1",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 7,
                    label: "Leaf2",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 8,
                    label: "Leaf3",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 9,
                    label: "Leaf4",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 10,
                    label: "Leaf5",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 11,
                    label: "h1a",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 12,
                    label: "h1b",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 13,
                    label: "h2a",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 14,
                    label: "h2b",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 15,
                    label: "h3a",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 16,
                    label: "h3b",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 17,
                    label: "h4a",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 18,
                    label: "h4b",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 19,
                    label: "h5a",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 20,
                    label: "h5b",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 21,
                    label: "h5c",
                    image: require('@/assets/img/switch.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 22,
                    label: "h6",
                    image: require('@/assets/img/server.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 23,
                    label: "h7",
                    image: require('@/assets/img/server.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },
                {
                    id: 24,
                    label: "h8",
                    image: require('@/assets/img/server.png'),
                    shape: 'image',
                    font: { color: '#ffffff' }
                },

            ]);

            // 边数据（初始为绿色）
            this.edges = new DataSet([
                {
                    id: 1,
                    from: 1,
                    to: 3,
                    width: 2,
                    color: { color: '#2063BA', highlight: '#2063BA' },
                    // label: '1Gbps'
                },
                {
                    id: 2,
                    from: 1,
                    to: 4,
                    width: 2,
                    color: { color: '#2063BA', highlight: '#2063BA' },
                    // label: '1Gbps'
                },
                {
                    id: 3,
                    from: 1,
                    to: 5,
                    width: 2,
                    color: { color: '#2063BA', highlight: '#2063BA' },
                    // label: '1Gbps'
                },
                {
                    id: 4,
                    from: 2,
                    to: 3,
                    width: 2,
                    color: { color: '#4BFA50', highlight: '#4BFA50' },
                    // label: '1Gbps'
                },
                {
                    id: 5,
                    from: 2,
                    to: 4,
                    width: 2,
                    color: { color: '#4BFA50', highlight: '#4BFA50' },
                    // label: '100Mbps'
                },
                {
                    id: 6,
                    from: 2,
                    to: 5,
                    width: 2,
                    color: { color: '#4BFA50', highlight: '#4BFA50' },
                    // label: '100Mbps'
                },
                {
                    id: 7,
                    from: 3,
                    to: 6,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 8,
                    from: 3,
                    to: 7,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 9,
                    from: 4,
                    to: 8,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 10,
                    from: 5,
                    to: 9,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 11,
                    from: 5,
                    to: 10,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 12,
                    from: 6,
                    to: 11,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 13,
                    from: 6,
                    to: 12,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 14,
                    from: 7,
                    to: 13,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 15,
                    from: 7,
                    to: 14,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 16,
                    from: 8,
                    to: 15,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 17,
                    from: 8,
                    to: 16,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 18,
                    from: 9,
                    to: 17,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 19,
                    from: 9,
                    to: 18,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 20,
                    from: 10,
                    to: 19,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 21,
                    from: 10,
                    to: 20,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 22,
                    from: 10,
                    to: 21,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 23,
                    from: 22,
                    to: 1,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 24,
                    from: 23,
                    to: 1,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
                {
                    id: 25,
                    from: 24,
                    to: 1,
                    width: 2,
                    color: { color: '#DCDCDC', highlight: '#DCDCDC' },
                    // label: '100Mbps'
                },
            ]);

            // 创建网络
            var container = document.getElementById("graph");
            var data = {
                nodes: this.nodes,
                edges: this.edges,
            };

            var options = {
                physics: {
                    stabilization: {
                        enabled: true,
                        iterations: 100
                    }
                },
                nodes: {
                    size: 30,
                    borderWidth: 2,
                    borderWidthSelected: 4,
                    color: {
                        border: '#2B7CE9',
                        background: '#97C2FC',
                        highlight: {
                            border: '#2B7CE9',
                            background: '#D2E5FF'
                        }
                    }
                },
                edges: {
                    smooth: {
                        type: 'continuous'
                    },
                    arrows: {
                        to: { enabled: false }
                    },
                    font: {
                        color: '#ffffff',
                        size: 12,
                        strokeWidth: 0
                    }
                }
            };

            this.network = new Network(container, data, options);
        },
        updateAttackStatus(isAttacked) {
            this.isAttacked = isAttacked;
            this.updateEdgeColors();
        },
        updateEdgeColors() {
            // 受影响边的ID
            const affectedEdges = [
                { id: 12, color: 'red' },
                { id: 7, color: 'red' },
                { id: 1, color: 'red' },
                { id: 4, color: 'red' },
                { id: 20, color: 'orange' },
                { id: 21, color: 'orange' },
                { id: 22, color: 'orange' },
                { id: 11, color: 'orange' },
                { id: 3, color: 'orange' },
                { id: 6, color: 'orange' },
            ];

            affectedEdges.forEach(edge => {
                this.edges.update({
                    id: edge.id,
                    color: {
                        color: this.isAttacked ? edge.color : '#DCDCDC',
                        highlight: this.isAttacked ? edge.color : '#DCDCDC'
                    },
                    width: this.isAttacked ? 4 : 2
                });
            });

            // 更新h1a节点状态
            this.nodes.update({
                id: 1,
                color: {
                    border: this.isAttacked ? '#FF0000' : '#2B7CE9',
                    background: this.isAttacked ? '#FFAAAA' : '#97C2FC'
                }
            });
        }
    }
};
</script>

<style scoped>
#graph {
    width: 100%;
    height: 100%;
    background-color: #1e1e1e;
    border-radius: 6px;
}
</style>