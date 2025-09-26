<template>
    <div class="hostPad">
        <el-button
      type="danger"
      class="attack-button"
      @click="simulateAttack"
      v-if="!isUnderAttack"
      style="opacity: 0; cursor: help;"
    >
      <el-icon><Warning /></el-icon>
      发起模拟攻击
    </el-button>
        <div id="left">
            <div class="title-container">
                <div class="title">
                    <el-icon><Monitor /></el-icon>
                    <span style="font-family: 等线;  font-size: 18px; font-weight: 700; color: white;">设备列表</span>
                </div>
                <el-tag type="info" size="small" style="color: white;">共 {{nodeData.length}} 台</el-tag>
            </div>
            <div class="titleback"><div class="triangle"></div></div>
            <el-card class="basicInfoListBox" shadow="hover">
                <topo-list
                    class="basicInfoListBox"
                    @updatehost="updatehost"
                    :nodeData="nodeData"
                    :type="0"
                ></topo-list>
            </el-card>
        </div>
        <div class="board">
            <el-card shadow="never" class="dashboard-card">
                <template #header>
                    <div class="card-header">
                        <el-icon><Warning /></el-icon>
                        <span>攻 击 监 控</span>
                        <el-tag type="danger" effect="dark" class="status-tag">
                            <el-icon><WarningFilled /></el-icon>
                            攻击进行中
                        </el-tag>
                    </div>
                </template>
                <div class="statistics">
                    <div class="flow" >
                        <el-card shadow="hover" class="chart-card">
                            <template #header>
                                <div class="chart-header">
                                    <span>流量监控</span>
                                    <el-select
                                        v-model="flowIndex"
                                        size="small"
                                        style="width: 120px"
                                        @change="handleFlowIndexChange"
                                    >
                                        <el-option label="实时流量" :value="0"></el-option>
                                        <el-option label="历史趋势" :value="1"></el-option>
                                    </el-select>
                                </div>
                            </template>
                            <div ref="chartContainer" style="width: 100%; height: 300px;"></div>
                        </el-card>
                    </div>

            <div class="total-attack-chart" v-if="showTotalAttackChart">
                        <el-card shadow="hover" class="chart-card">
                        <template #header>
                            <div class="chart-header">
                            <span>总攻击量统计</span>
                            </div>
                        </template>
                        <div ref="totalAttackChartContainer" style="width: 100%; height: 250px;"></div>
                        </el-card>
                    </div>

                    <div class="metrics-grid">
                        <el-card shadow="hover" class="metric-card">
                            <div class="metric-content">
                                <div class="metric-icon">
                                    <el-icon><Cpu /></el-icon>
                                </div>
                                <div class="metric-info">
                                    <div class="metric-title">服务器CPU使用率</div>
                                    <div class="metric-value">95%</div>
                                </div>
                            </div>
                        </el-card>
                        <el-card shadow="hover" class="metric-card">
                            <div class="metric-content">
                                <div class="metric-icon">
                                    <el-icon><Memory /></el-icon>
                                </div>
                                <div class="metric-info">
                                    <div class="metric-title">服务器内存使用率</div>
                                    <div class="metric-value">90%</div>
                                </div>
                            </div>
                        </el-card>
                        <el-card shadow="hover" class="metric-card">
                            <div class="metric-content">
                                <div class="metric-icon">
                                    <el-icon><Connection /></el-icon>
                                </div>
                                <div class="metric-info">
                                    <div class="metric-title">网络延迟增加</div>
                                    <div class="metric-value">500ms</div>
                                </div>
                            </div>
                        </el-card>
                        <el-card shadow="hover" class="metric-card">
                            <div class="metric-content">
                                <div class="metric-icon">
                                    <el-icon><Clock /></el-icon>
                                </div>
                                <div class="metric-info">
                                    <div class="metric-title">警报次数</div>
                                    <div class="metric-value">50次/分钟</div>
                                </div>
                            </div>
                        </el-card>
                    </div>
                </div>
            </el-card>
        </div>
    </div>
</template>

<script>
import TopoList from '@/components/TopoList.vue'
import * as echarts from 'echarts'
import {
  Monitor, Warning, WarningFilled, SuccessFilled, InfoFilled,
  Cpu, Memory, Connection, Clock
} from '@element-plus/icons-vue'

export default {
  name: 'HostMessage',
  components: {
    TopoList,
    Monitor, Warning, WarningFilled, SuccessFilled, InfoFilled,
    Cpu, Memory, Connection, Clock
  },
  data() {
    return {
        showTotalAttackChart: false,
      totalAttackChartInstance: null,
      totalAttackData: [
        { name: 'SYN Flood', value: 0 },
        { name: '带宽耗尽', value: 0 },
        { name: '慢连接', value: 0 }
      ],
      isUnderAttack: false,
      currentAttackType: null,
      selectedHost: null,
      flowIndex: 0,
      nodeData: [
        { list: 1, id: 'h1', ip: '未受攻击' },
        { list: 2, id: 'h2', ip: '未受攻击' },
        { list: 3, id: 'h3', ip: '未受攻击' },
        { list: 4, id: 'h4', ip: '未受攻击' }
      ],
      attackData: {
        synFlood: this.generateAttackData('syn'),
        bandwidth: this.generateAttackData('bandwidth'),
        slowConnect: this.generateAttackData('slow')
      },
      chartInstance: null,
      state: 0
    }
  },
  methods: {
    generateAttackData(type) {
      const data = [];
      const baseTime = new Date(2024, 0, 1, 10, 34, 21);

      for (let i = 0; i < 16; i++) {
        let value;
        switch(type) {
          case 'syn':
            // SYN Flood攻击特征：突发高流量
            value = Math.floor(Math.random() * 35 + 5);
            if (i > 5 && i < 10) value += 40; // 模拟攻击峰值
            break;
          case 'bandwidth':
            // 带宽耗尽攻击特征：持续高流量波动
            value = Math.floor(Math.sin(i/2) * 400 + 600);
            break;
          case 'slow':
            // 慢连接攻击特征：缓慢上升
            value = Math.floor(i * 80 + 200);
            break;
          default:
            value = 0;
        }
        data.push({
          time: new Date(baseTime.getTime() + i * 1000),
          value: value
        });
      }
      return data;
    },

    simulateAttack() {
      this.isUnderAttack = true;
      this.state = 1; // 设置为攻击状态

      // 替换h2为三个攻击项
      this.nodeData = [
        { list: 1, id: 'h2a', ip: '未受攻击' },
        { list: 2, id: 'h1a', ip: 'SYN Flood攻击' },
        { list: 3, id: 'h1b', ip: '未受攻击' },
        { list: 4, id: 'h2b', ip: '未受攻击' },
        { list: 5, id: 'h3', ip: '未受攻击' },
        { list: 6, id: 'h4', ip: '未受攻击' }
      ];
    },

    updatehost(e) {
      const selected = this.nodeData.find(item => item.list === e);
      if (selected.ip.includes('攻击')) {
        this.currentAttackType = this.getAttackType(selected.ip);
        this.updateChart();
      } else {
        this.selectedHost = null;
        if (this.chartInstance) {
          this.chartInstance.clear();
        }
      }
    },

    getAttackType(ip) {
      if(ip.includes('SYN')) return 'synFlood';
      if(ip.includes('带宽')) return 'bandwidth';
      if(ip.includes('慢连接')) return 'slowConnect';
      return null;
    },

    initChart() {
      if (!this.chartInstance) {
        this.chartInstance = echarts.init(this.$refs.chartContainer);
        window.addEventListener('resize', this.resizeChart);
      }
      this.updateChart();
    },

    updateChart() {
      if (!this.currentAttackType || !this.chartInstance) return;

      const attackData = this.attackData[this.currentAttackType];
      const option = {
        backgroundColor: '#1E1E1E',
        tooltip: {
          trigger: 'axis',
          formatter: params => {
            const time = params[0].data.time;
            return `时间: ${time.getMinutes()}:${time.getSeconds().toString().padStart(2, '0')}<br/>流量: ${params[0].data.value} Mbps`;
          }
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'category',
          data: attackData.map(d => `${d.time.getSeconds()}`),
          axisLine: {
            lineStyle: { color: '#6ec8ff' }
          },
          axisLabel: {
            color: '#a0d8ff',
            formatter: value => `${value}s`
          }
        },
        yAxis: {
          type: 'value',
          name: '流量 (Mbps)',
          nameTextStyle: { color: '#a0d8ff' },
          axisLine: { lineStyle: { color: '#6ec8ff' } },
          axisLabel: { color: '#a0d8ff' },
          splitLine: {
            lineStyle: { color: 'rgba(100, 180, 255, 0.1)' }
          }
        },
        series: [{
          data: attackData,
          type: 'line',
          smooth: true,
          lineStyle: {
            width: 3,
            color: this.getLineColor(this.currentAttackType)
          },
          itemStyle: {
            color: this.getLineColor(this.currentAttackType)
          },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: this.getAreaColor(this.currentAttackType) },
              { offset: 1, color: 'rgba(110, 200, 255, 0.1)' }
            ])
          }
        }]
      };

      this.chartInstance.setOption(option);
    },

    getLineColor(type) {
      const colors = {
        synFlood: '#ff4757',
        bandwidth: '#2ed573',
        slowConnect: '#ffa502'
      };
      return colors[type] || '#6ec8ff';
    },

    getAreaColor(type) {
      const colors = {
        synFlood: 'rgba(255, 71, 87, 0.5)',
        bandwidth: 'rgba(46, 213, 115, 0.5)',
        slowConnect: 'rgba(255, 165, 2, 0.5)'
      };
      return colors[type] || 'rgba(110, 200, 255, 0.5)';
    },

    resizeChart() {
      this.chartInstance && this.chartInstance.resize();
    },

    handleFlowIndexChange() {
      this.updateChart();
    }
  },
  mounted() {
    this.$nextTick(() => {
      this.initChart();
    });
  },
  beforeUnmount() {
    if (this.chartInstance) {
      window.removeEventListener('resize', this.resizeChart);
      this.chartInstance.dispose();
    }
  }
}
</script>

<style scoped>
/* 样式部分保持不变 */
.hostPad {
    width: 100vw;
    height: 100vh;
    position: relative;
    background: radial-gradient(circle at center, #0a0a1a 0%, #000000 100%);
    display: flex;
    padding: 20px;
    box-sizing: border-box;
}

#left {
    margin-top: 1vh;
    width: 25vw;
    height: 90vh;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.title-container {
    position: relative;
    right: 50px;
    width: 80%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.title {
    display: flex;
    align-items: center;
    color: #6ec8ff;
    font-size: 16px;
    font-weight: 500;
}

.title span {
    margin-left: 8px;
}

.basicInfoListBox {
    padding: 1vh;
    width: 100%;
    height: 75vh;
    overflow-x: hidden;
    overflow-y: scroll;
    scrollbar-width: none;
    background-color: rgba(10, 15, 30, 0.8);
    border: 1px solid rgba(100, 180, 255, 0.2);
    border-radius: 8px;
}

.titleback{
    position: relative;
    right: 40px;
    width: 80%;
    height: 2vh;
    margin-bottom: 2vh;
    background: linear-gradient(to right,
        rgba(80, 160, 255, 0.8) 0%,
        rgba(120, 100, 255, 0.6) 50%,
        rgba(80, 160, 255, 0.1) 100%);
    border-radius: 4px;
}

.triangle {
    width: 0;
    height: 0;
    border-top: 0.75vh solid rgba(80, 160, 255, 0.8);
    border-left: 3vh solid rgba(80, 160, 255, 0.8);
    border-right: 3vh solid transparent;
    border-bottom: 0.75vh solid transparent;
}

.board {
    flex: 1;
    margin-left: 20px;
    height: 90vh;
}

.dashboard-card {
    height: 100%;
    background-color: rgba(10, 15, 30, 0.8);
    border: 1px solid rgba(100, 180, 255, 0.2);
    color: #c0e4ff;
    border-radius: 8px;
}

.dashboard-card :deep(.el-card__header) {
    background: linear-gradient(to right,
        rgba(80, 160, 255, 0.2) 0%,
        rgba(120, 100, 255, 0.15) 100%);
    border-bottom: 1px solid rgba(100, 180, 255, 0.2);
    border-radius: 8px 8px 0 0;
}

.card-header {
    display: flex;
    align-items: center;
    font-size: 18px;
    font-weight: 500;
    color: #6ec8ff;
}

.card-header .el-icon {
    margin-right: 10px;
    font-size: 20px;
    color: #6ec8ff;
}

.status-tag {
    margin-left: 15px;
    border-radius: 12px;
}

.statistics {
    height: calc(100% - 60px);
    display: flex;
    flex-direction: column;
}

.flow {
    height: 80%;
}

.chart-card {
    height: 100%;
    background-color: transparent;
    border: 1px solid rgba(100, 180, 255, 0.2);
    border-radius: 8px;
}

.chart-card :deep(.el-card__header) {
    padding: 12px 20px;
    background: linear-gradient(to right,
        rgba(80, 160, 255, 0.1) 0%,
        rgba(120, 100, 255, 0.08) 100%);
    border-bottom: 1px solid rgba(100, 180, 255, 0.2);
    border-radius: 8px 8px 0 0;
}

.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #6ec8ff;
}

.metrics-grid {
    margin-top: 40px;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    height: 16%;
}

.metric-card {
    background-color: rgba(15, 25, 50, 0.6);
    border: 1px solid rgba(100, 180, 255, 0.2);
    border-radius: 8px;
    padding: 12px;
}

.metric-content {
    display: flex;
    align-items: center;
    height: 100%;
}

.metric-icon {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg,
        rgba(80, 160, 255, 0.3) 0%,
        rgba(120, 100, 255, 0.3) 100%);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 12px;
}

.metric-icon .el-icon {
    font-size: 18px;
    color: #a0d8ff;
}

.metric-info {
    flex: 1;
}

.metric-title {
    font-size: 13px;
    color: #a0d8ff;
    margin-bottom: 4px;
}

.metric-value {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
}

/* 标签颜色调整 */
.el-tag--info {
    background-color: rgba(80, 160, 255, 0.2);
    border-color: rgba(80, 160, 255, 0.3);
    color: #6ec8ff;
}

.el-tag--danger {
    background-color: rgba(255, 80, 120, 0.2);
    border-color: rgba(255, 80, 120, 0.3);
    color: #ff6e8b;
}

.el-tag--success {
    background-color: rgba(80, 255, 160, 0.2);
    border-color: rgba(80, 255, 160, 0.3);
    color: #6effb3;
}

/* 下拉选择框样式 */
.el-select {
    --el-select-border-color-hover: rgba(100, 180, 255, 0.5);
    --el-select-input-focus-border-color: rgba(100, 180, 255, 0.8);
}

.el-select-dropdown {
    background-color: #0a0a1a;
    border: 1px solid rgba(100, 180, 255, 0.3);
}

.el-select-dropdown__item {
    color: #c0e4ff;
}

.el-select-dropdown__item.hover {
    background-color: rgba(80, 160, 255, 0.2);
}

/* 添加攻击按钮样式 */
.attack-button {
  position: absolute;
  right: 30px;
  top: 30px;
  z-index: 1000;
  background: linear-gradient(45deg, #ff6b6b, #ff4757);
  border: none;
  color: white;
  font-weight: bold;
  box-shadow: 0 4px 6px rgba(255, 107, 107, 0.3);
  transition: all 0.3s ease;
}

.attack-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 8px rgba(255, 107, 107, 0.4);
}

.attack-button:active {
  transform: translateY(0);
}

/* 调整图表容器高度 */
.chart-card {
  height: 320px !important;
}

/* 攻击状态标签样式 */
.attack-tag {
  margin-left: 8px;
  background: rgba(255, 71, 87, 0.2);
  border-color: rgba(255, 71, 87, 0.3);
  color: #ff4757;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .attack-button {
    right: 15px;
    top: 15px;
    padding: 8px 12px;
    font-size: 12px;
  }
}
.total-attack-chart {
  margin-top: 20px;
  transition: all 0.3s ease;
}

.total-attack-chart .chart-card {
  height: 250px !important;
  background-color: rgba(10, 15, 30, 0.8);
  border: 1px solid rgba(100, 180, 255, 0.2);
}

/* 调整原有布局 */
.statistics {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.metrics-grid {
  margin-top: 0;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .total-attack-chart .chart-card {
    height: 200px !important;
  }
}
</style>