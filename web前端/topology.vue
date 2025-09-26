<template>
    <div id="view">
        <div id="left-charts" style="padding-bottom: 20px;">
            <div class="attack-button">
                <el-button type="danger" @click="simulateAttack" style="opacity: 0; cursor: help;">模拟h1a遭受攻击</el-button>
            </div>
            <div class="left-block" style="margin-bottom: 20px; height: 40%;">
                <!-- 新增的链路状态列表 -->
                <div class="header" >
                    <div class="title">链路状态</div>
                    <div class="count-badge">{{ linkData.length + 3 }}条链路</div>
                </div>
                <div class="title-background">
                    <div class="triangle"></div>
                </div>
                <div id="link-chart" >
                    <topo-list :nodeData="this.linkData" :type="1"></topo-list>
                </div>
            </div>
        <div class="left-block2" >
            <div class="header">
                <div class="title">设备列表</div>
                <div class="count-badge">{{ nodeData.length + 3}}台设备</div>
            </div>
            <div class="title-background">
                <div class="triangle"></div>
            </div>
            <div id="device-chart" ref="device_chart">
                <topo-list :nodeData="this.nodeData" :type="1"></topo-list>
            </div>
         </div>
        </div>
        <div class="graph-container">
            <div class="graph-title">网络拓扑图</div>
            <net-visual ref="netVisual"></net-visual>
        </div>
    </div>
</template>

<script>
import NetVisual from '@/components/NetVisual.vue'
import TopoList from '@/components/TopoList.vue'
import { ElMessageBox, ElButton } from 'element-plus'

export default {
    data(){
        return{
            nodeData: [
                { id: 's1', ip: '192.168.1.1' },
                { id: 's2', ip: '192.168.1.2' },
                { id: 's3', ip: '192.168.1.3' },
                { id: 'h1a', ip: '10.0.0.1' },
                { id: 'h1b', ip: '10.0.0.2' },
                { id: 'h2a', ip: '10.0.0.3' },
                { id: 'h2b', ip: '10.0.0.4' },
                { id: 'h3a', ip: '10.0.0.5' },
                { id: 'h3b', ip: '10.0.0.6' },
                { id: 'h4a', ip: '10.0.0.7' },
                { id: 'h4b', ip: '10.0.0.8' },
                // { id: 'h5a', ip: '10.0.0.9' },
                // { id: 'h5b', ip: '10.0.0.10'},
                // { id: 'h5c', ip: '10.0.0.11'},
            ],
            linkData: [
                { id: 's1-h1a', ip: '1Gbps' },
                { id: 's1-h1b', ip: '1Gbps' },
                { id: 's2-h2a', ip: '1Gbps' },
                { id: 's2-h2b', ip: '1Gbps' },
                { id: 's3-h3a', ip: '1Gbps' },
                { id: 's3-h3b', ip: '1Gbps' },
                { id: 's4-h4a', ip: '1Gbps' },
                // { id: 's4-h4b', ip: '1Gbps' },
                // { id: 's5-h5a', ip: '1Gbps' },
                // { id: 's5-h5b', ip: '1Gbps' },
            ],
            isAttacked: false
        }
    },
    components: {
        TopoList,
        NetVisual,
        ElButton
    },
    methods: {
        simulateAttack() {
            this.isAttacked = !this.isAttacked;
            this.$refs.netVisual.updateAttackStatus(this.isAttacked);

            if (this.isAttacked) {
                ElMessageBox.alert(
                    'h1a受到网络攻击！',
                    '攻击警告',
                    {
                        type: 'error',
                        confirmButtonText: '确定',
                    }
                )
            } else {
                ElMessageBox.alert(
                    'h1a攻击已解除，链路恢复正常',
                    '恢复通知',
                    {
                        type: 'success',
                        confirmButtonText: '确定',
                    }
                )
            }
        },
        changeState() {
            // 保留原有状态变更逻辑（可选）
        }
    },
    mounted() {
        this.$refs.device_chart.scrollTop = 0;
    },
}
</script>

<style scoped>
@font-face
{
font-family: myFont;
src: url('../assets/img/font/ZhengQingKeLengKuTi-2.ttf'),
}

#view {
    height: 100vh;
    width: 100%;
    display: flex;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    /* border-radius: 8px; */
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

#left-charts {
    height: 90%;
    width: 30%;
    display: flex;
    align-items: center;
    flex-direction: column;
    justify-content: space-around;
    padding: 20px;
    background: #121212;
    border-radius: 8px 0 0 8px;
    box-shadow: 2px 0 10px rgba(0, 0, 0, 0.4);
    border-right: 1px solid #333;
}

.left-block {
    height: 100%;
    width: 100%;
    display: flex;
    flex-direction: column;
    background: #1e1e1e;
    border-radius: 6px;
    box-shadow: 0 2px 15px rgba(0, 0, 0, 0.3);
    border: 1px solid #2a2a2a;
}
.left-block2 {
    height: 100%;
    width: 100%;
    display: flex;
    flex-direction: column;
    background: #1e1e1e;
    border-radius: 6px;
    box-shadow: 0 2px 15px rgba(0, 0, 0, 0.3);
    border: 1px solid #2a2a2a;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 15px 20px 0;
    border-bottom-left-radius: 5px;
}

.title {
    font-size: 18px;
    font-weight: 600;
    color: #e0e0e0;
    letter-spacing: 0.5px;
    text-shadow: 0 0 5px rgba(100, 149, 237, 0.5);
}

.count-badge {
    background: linear-gradient(135deg, #4a6bff 0%, #8a2be2 100%);
    color: white;
    padding: 5px 15px;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 500;
    box-shadow: 0 0 8px rgba(74, 107, 255, 0.5);
    position: relative;
    font-family: 等线;
    right: 30px;
}

.title-background {
    width: 100%;
    height: 4px;
    margin: 10px 0;
    background: linear-gradient(to right, #4a6bff, #8a2be2, rgba(138, 43, 226, 0.3));
    position: relative;
}

.triangle {
    position: absolute;
    left: 0;
    top: -4px;
    width: 0;
    height: 0;
    border-left: 8px solid #4a6bff;
    /* border-top: 4px solid transparent; */
    /* border-bottom: 4px solid transparent; */
    filter: drop-shadow(0 0 3px #4a6bff);
}

#device-chart {
    display: flex;
    flex-direction: column;
    height: calc(100% - 60px);
    width: 100%;
    padding: 0 15px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #4a6bff #2a2a2a;
}

#device-chart::-webkit-scrollbar {
    width: 6px;
}

#device-chart::-webkit-scrollbar-track {
    background: #2a2a2a;
    border-radius: 3px;
}

#device-chart::-webkit-scrollbar-thumb {
    background: linear-gradient(#4a6bff, #8a2be2);
    border-radius: 3px;
    box-shadow: inset 0 0 3px rgba(255, 255, 255, 0.2);
}

.graph-container {
    width: 70%;
    height: 84%;
    padding: 20px;
    background: #121212;
    border-radius: 0 8px 8px 0;
    box-shadow: -2px 0 10px rgba(0, 0, 0, 0.4);
    border-left: 1px solid #333;
}

#right-graph {
    width: 100%;
    height: 100%;
    border: 1px solid #333;
    border-radius: 6px;
    box-shadow: inset 0 0 15px rgba(0, 0, 0, 0.5), 0 0 15px rgba(74, 107, 255, 0.2);
    background: #1e1e1e;
}

#device-chart, #link-chart {
    display: flex;
    flex-direction: column;
    width: 100%;
    padding: 0 15px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #4a6bff #2a2a2a;
}

#device-chart::-webkit-scrollbar,
#link-chart::-webkit-scrollbar {
    width: 6px;
}

#device-chart::-webkit-scrollbar-track,
#link-chart::-webkit-scrollbar-track {
    background: #2a2a2a;
    border-radius: 3px;
}

#device-chart::-webkit-scrollbar-thumb,
#link-chart::-webkit-scrollbar-thumb {
    background: linear-gradient(#4a6bff, #8a2be2);
    border-radius: 3px;
    box-shadow: inset 0 0 3px rgba(255, 255, 255, 0.2);
}

.graph-title {
    font-size: 26px;
    font-weight: 600;
    font-family: myFont;
    text-align: center;
    padding: 6px 0;
    margin-bottom: 15px;
    color: #ffffff; /* 白色文字提高可读性 */
    background: linear-gradient(90deg,
        rgba(0, 0, 0, 0.8) 0%,
        rgba(0, 60, 120, 0.8) 50%,
        rgba(90, 30, 150, 0.8) 100%); /* 黑色-科技蓝-科技紫渐变 */
    border: 1px solid rgba(0, 123, 255, 0.3); /* 微妙的蓝色边框 */
    box-shadow:
        0 2px 8px rgba(0, 123, 255, 0.3), /* 外发光效果 */
        inset 0 0 10px rgba(90, 30, 150, 0.5); /* 内发光效果 */
    letter-spacing: 1px;
    text-shadow: 0 0 5px rgba(0, 123, 255, 0.5); /* 文字微光效果 */
    border-radius: 4px; /* 轻微圆角 */
    position: relative;
    overflow: hidden;
}
</style>