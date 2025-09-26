<template>
  <div class="traffic-control-panel">
    <!-- 左侧控制区域 -->
    <div class="control-panel">
      <h3 style="color: white;">清洗流量</h3>

      <!-- 粒度选择 Card -->
      <div class="control-card">
        <div class="card-header">
          <h4>流量粒度控制</h4>
        </div>
        <div class="card-body">
          <div class="control-group">
            <label>选择粒度</label>
            <div class="radio-group">
              <label>
                <input type="radio" v-model="granularity" value="coarse"> 粗粒度
              </label>
              <label>
                <input type="radio" v-model="granularity" value="fine"> 细粒度
              </label>
            </div>
          </div>
        </div>
      </div>

      <!-- IP控制 Card -->
      <div class="control-card">
        <div class="card-header">
          <h4>IP控制</h4>
        </div>
        <div class="card-body">
          <div class="control-group">
            <label>选择IP</label>
            <input type="text" v-model="selectedIP" placeholder="输入IPv6地址" style="width: 95%;">
          </div>
          <div class="button-group">
            <button class="allow-btn" @click="handleAllow">放行</button>
            <button class="block-btn" @click="handleBlock">屏蔽</button>
          </div>
        </div>
      </div>

      <!-- 流量类型 Card -->
      <div class="control-card">
        <div class="card-header">
          <h4>流量类型</h4>
        </div>
        <div class="card-body">
          <ul class="traffic-types">
            <li v-for="type in trafficTypes" :key="type.name" :style="{color: type.color}">
              <input type="checkbox" v-model="type.visible"> {{type.name}}
            </li>
          </ul>
        </div>
      </div>
    </div>

    <!-- 中间图表区域 -->
    <div class="chart-panel" style="position: relative; bottom: 10px;">
      <div class="chart-container">
        <canvas ref="chartCanvas"></canvas>
      </div>
    </div>

    <!-- 右侧IP列表区域 -->
    <div class="ip-list-panel">
      <div class="traffic-category" v-for="category in trafficCategories" :key="category.name">
        <div class="category-header" :style="{ color: category.color }">
          <h3>{{ category.name }}</h3>
          <span class="status">{{ category.status }}</span>
        </div>
        <ul class="ip-list">
          <li v-for="ip in category.ips" :key="ip.address">
            <span class="ip-address">{{ ip.address }}</span>
            <span class="ip-status" :class="ip.statusClass">{{ ip.status }}</span>
          </li>
        </ul>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, watch } from 'vue';
import Chart from 'chart.js/auto';

export default {
  name: 'TrafficControlPanel',
  setup() {
    // 数据状态
    const granularity = ref('coarse');
    const selectedIP = ref('');
    const trafficData = ref([]);
    const chartInstance = ref(null);
    const chartCanvas = ref(null);

    // 预设数据 - 更加平稳
    const presetData = [
      { time: '11:06:17', 'SYN Flood': 58, 'Exhaustion Attack': 22, 'Bandwidth': 24, 'Flower': 20 },
      { time: '11:06:18', 'SYN Flood': 59, 'Exhaustion Attack': 21, 'Bandwidth': 23, 'Flower': 21 },
      { time: '11:06:19', 'SYN Flood': 57, 'Exhaustion Attack': 23, 'Bandwidth': 25, 'Flower': 22 },
      { time: '11:06:20', 'SYN Flood': 58, 'Exhaustion Attack': 22, 'Bandwidth': 24, 'Flower': 20 },
      { time: '11:06:21', 'SYN Flood': 60, 'Exhaustion Attack': 21, 'Bandwidth': 23, 'Flower': 21 },
      { time: '11:06:22', 'SYN Flood': 59, 'Exhaustion Attack': 22, 'Bandwidth': 25, 'Flower': 22 },
      { time: '11:06:23', 'SYN Flood': 58, 'Exhaustion Attack': 23, 'Bandwidth': 24, 'Flower': 20 },
      { time: '11:06:24', 'SYN Flood': 57, 'Exhaustion Attack': 22, 'Bandwidth': 23, 'Flower': 21 },
      { time: '11:06:25', 'SYN Flood': 59, 'Exhaustion Attack': 21, 'Bandwidth': 25, 'Flower': 22 }
    ];

    // 流量类型配置
    const trafficTypes = ref([
      { name: 'SYN Flood', color: '#4CAF50', visible: true },
      { name: 'Exhaustion Attack', color: '#F44336', visible: true },
      { name: 'Bandwidth', color: '#2196F3', visible: true },
      { name: 'Flower', color: '#FFC107', visible: true }
    ]);

    // 流量分类及IP列表
    const trafficCategories = ref([
      {
        name: 'SYN Flood',
        color: '#4CAF50',
        status: '已加载',
        ips: [
          { address: 'ADA7:5BD4:25DB:E125:A123:65FB:CF31:F5BC/128', status: '已屏蔽', statusClass: 'blocked' },
          { address: 'BEEF:CAFE:1234:5678:90AB:CDEF:0123:4567/128', status: '已加载', statusClass: 'loaded' },
          { address: 'DEAD:BEEF:0000:0000:0000:0000:0000:0001/128', status: '已屏蔽', statusClass: 'blocked' }
        ]
      },
      {
        name: 'Exhaustion Attack',
        color: '#F44336',
        status: '已屏蔽',
        ips: [
          { address: '2001:0DB8:AC10:FE01:0000:0000:0000:0000/128', status: '已屏蔽', statusClass: 'blocked' },
          { address: '2001:0DB8:AC10:FE01:0000:0000:0000:0001/128', status: '已屏蔽', statusClass: 'blocked' }
        ]
      },
      {
        name: 'Bandwidth',
        color: '#2196F3',
        status: '已加载',
        ips: [
          { address: 'FE80:0000:0000:0000:0202:B3FF:FE1E:8329/128', status: '已加载', statusClass: 'loaded' },
          { address: 'FE80:0000:0000:0000:0202:B3FF:FE1E:8330/128', status: '已加载', statusClass: 'loaded' },
          { address: 'FE80:0000:0000:0000:0202:B3FF:FE1E:8331/128', status: '已屏蔽', statusClass: 'blocked' }
        ]
      },
      {
        name: 'Flower',
        color: '#FFC107',
        status: '已加载',
        ips: [
          { address: 'FF02:0000:0000:0000:0000:0000:0000:0001/128', status: '已加载', statusClass: 'loaded' },
          { address: 'FF02:0000:0000:0000:0000:0000:0000:0002/128', status: '已屏蔽', statusClass: 'blocked' },
          { address: 'FF02:0000:0000:0000:0000:0000:0000:0003/128', status: '已加载', statusClass: 'loaded' },
          { address: 'FF02:0000:0000:0000:0000:0000:0000:0004/128', status: '已屏蔽', statusClass: 'blocked' }
        ]
      }
    ]);

    // 初始化数据
    trafficData.value = [...presetData];

    // 初始化图表
    const initChart = () => {
      if (chartInstance.value) {
        chartInstance.value.destroy();
      }

      const ctx = chartCanvas.value.getContext('2d');

      // 准备数据集
      const datasets = trafficTypes.value
        .filter(type => type.visible)
        .map(type => ({
          label: type.name,
          data: trafficData.value.map(item => item[type.name]),
          borderColor: type.color,
          backgroundColor: 'rgba(0, 0, 0, 0)',
          borderWidth: 2,
          tension: 0.1,
          fill: false,
          pointRadius: 3,
          pointHoverRadius: 5
        }));

      chartInstance.value = new Chart(ctx, {
        type: 'line',
        data: {
          labels: trafficData.value.map(item => item.time),
          datasets: datasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: false,
              min: 0,
              max: 70,
              title: {
                display: true,
                text: '流量',
                color: '#e6f1ff'
              },
              grid: {
                color: 'rgba(255, 255, 255, 0.1)'
              },
              ticks: {
                color: '#e6f1ff'
              }
            },
            x: {
              title: {
                display: true,
                text: '时间',
                color: '#e6f1ff'
              },
              grid: {
                color: 'rgba(255, 255, 255, 0.1)'
              },
              ticks: {
                color: '#e6f1ff'
              }
            }
          },
          plugins: {
            legend: {
              position: 'top',
              labels: {
                color: '#e6f1ff',
                font: {
                  size: 12
                }
              }
            },
            tooltip: {
              mode: 'index',
              intersect: false,
              backgroundColor: 'rgba(0, 0, 0, 0.7)',
              titleColor: '#64ffda',
              bodyColor: '#e6f1ff',
              borderColor: '#1e2a4a',
              borderWidth: 1
            }
          },
          interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
          },
          animation: {
            duration: 1000
          }
        }
      });
    };

    // 处理放行操作
    const handleAllow = () => {
      if (!selectedIP.value) {
        alert('请先输入IP地址');
        return;
      }

      // 检查IP是否在SYN Flood列表中
      const synFloodCategory = trafficCategories.value.find(c => c.name === 'SYN Flood');
      const matchedIP = synFloodCategory.ips.find(ip => ip.address === selectedIP.value);

      if (matchedIP) {
        // 更新IP状态
        matchedIP.status = '已加载';
        matchedIP.statusClass = 'loaded';

        // 模拟流量下降效果
        const newData = JSON.parse(JSON.stringify(trafficData.value));
        const lastValue = newData[newData.length - 1]['SYN Flood'];

        // 添加下降趋势数据点
        for (let i = 0; i < 5; i++) {
          const newTime = new Date();
          newTime.setSeconds(newTime.getSeconds() - (4 - i));
          const timeStr = `${newTime.getHours().toString().padStart(2, '0')}:${newTime.getMinutes().toString().padStart(2, '0')}:${newTime.getSeconds().toString().padStart(2, '0')}`;

          newData.push({
            time: timeStr,
            'SYN Flood': Math.max(20, lastValue - (i + 1) * 8),
            'Exhaustion Attack': 22 + Math.sin(i) * 2,
            'Bandwidth': 24 + Math.cos(i) * 1,
            'Flower': 20 + Math.sin(i * 0.5) * 1
          });
        }

        trafficData.value = newData;
        alert(`已放行IP: ${selectedIP.value}`);
      } else {
        alert('输入的IP地址不匹配任何SYN Flood记录');
      }
    };

    // 处理屏蔽操作
    const handleBlock = () => {
      if (!selectedIP.value) {
        alert('请先输入IP地址');
        return;
      }

      // 检查IP是否在SYN Flood列表中
      const synFloodCategory = trafficCategories.value.find(c => c.name === 'SYN Flood');
      const matchedIP = synFloodCategory.ips.find(ip => ip.address === selectedIP.value);

      if (matchedIP) {
        // 更新IP状态
        matchedIP.status = '已屏蔽';
        matchedIP.statusClass = 'blocked';

        // 模拟流量上升效果
        const newData = JSON.parse(JSON.stringify(trafficData.value));
        const lastValue = newData[newData.length - 1]['SYN Flood'];

        // 添加上升趋势数据点
        for (let i = 0; i < 5; i++) {
          const newTime = new Date();
          newTime.setSeconds(newTime.getSeconds() - (4 - i));
          const timeStr = `${newTime.getHours().toString().padStart(2, '0')}:${newTime.getMinutes().toString().padStart(2, '0')}:${newTime.getSeconds().toString().padStart(2, '0')}`;

          newData.push({
            time: timeStr,
            'SYN Flood': Math.min(80, lastValue + (i + 1) * 8),
            'Exhaustion Attack': 22 + Math.sin(i) * 2,
            'Bandwidth': 24 + Math.cos(i) * 1,
            'Flower': 20 + Math.sin(i * 0.5) * 1
          });
        }

        trafficData.value = newData;
        alert(`已屏蔽IP: ${selectedIP.value}`);
      } else {
        alert('输入的IP地址不匹配任何SYN Flood记录');
      }
    };

    // 监听粒度变化
    watch(granularity, () => {
      trafficData.value = [...presetData];
    });

    // 监听流量类型可见性变化
    watch(trafficTypes, () => {
      if (chartInstance.value) {
        initChart();
      }
    }, { deep: true });

    // 监听数据变化
    watch(trafficData, () => {
      if (chartInstance.value) {
        initChart();
      }
    }, { deep: true });

    // 组件挂载时初始化图表
    onMounted(() => {
      initChart();
    });

    return {
      granularity,
      selectedIP,
      trafficTypes,
      trafficCategories,
      chartCanvas,
      handleAllow,
      handleBlock
    };
  }
};
</script>

<style scoped>
.traffic-control-panel {
  display: flex;
  height: 100vh;
  background-color: #0a192f; /* 深蓝色背景 */
  color: #e6f1ff;
  font-family: Arial, sans-serif;
}

.control-panel {
  width: 300px;
  padding: 20px;
  background-color: #112240;
  border-right: 1px solid #1e2a4a;
}

.chart-panel {
  flex: 1;
  padding: 20px;
}

.ip-list-panel {
  width: 400px;
  padding: 20px;
  background-color: #112240;
  border-left: 1px solid #1e2a4a;
  overflow-y: auto;
}

.chart-container {
  height: 100%;
  position: relative;
}

.control-group {
  margin-bottom: 20px;
}

.control-group label {
  display: block;
  margin-bottom: 8px;
  font-weight: bold;
  color: #e6f1ff;
}

.control-group input[type="text"] {
  width: 100%;
  padding: 8px;
  background-color: #0a192f;
  border: 1px solid #1e2a4a;
  color: #e6f1ff;
  border-radius: 4px;
}

.radio-group {
  display: flex;
  gap: 15px;
}

.radio-group label {
  display: flex;
  align-items: center;
  gap: 5px;
  font-weight: normal;
  color: #e6f1ff;
}

.button-group {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.button-group button {
  flex: 1;
  padding: 10px;
  border: none;
  border-radius: 4px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
}

.button-group button:hover {
  opacity: 0.9;
  transform: translateY(-1px);
}

.allow-btn {
  background-color: #4CAF50; /* 绿色放行按钮 */
  color: white;
}

.block-btn {
  background-color: #F44336; /* 红色屏蔽按钮 */
  color: white;
}

.traffic-types ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.traffic-types li {
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* IP列表样式 */
.traffic-category {
  margin-bottom: 20px;
}

.category-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  padding-bottom: 5px;
  border-bottom: 1px solid #1e2a4a;
}

.category-header h3 {
  margin: 0;
  font-size: 16px;
}

.status {
  font-size: 12px;
  opacity: 0.8;
}

.ip-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.ip-list li {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid #1e2a4a;
  font-size: 14px;
}

.ip-address {
  font-family: monospace;
  color: #e6f1ff;
}

.ip-status {
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 12px;
  font-weight: bold;
}

.ip-status.loaded {
  background-color: rgba(76, 175, 80, 0.2);
  color: #4CAF50;
}

.ip-status.blocked {
  background-color: rgba(244, 67, 54, 0.2);
  color: #F44336;
}

h3, h4 {
  color: #64ffda;
  margin-top: 0;
  margin-bottom: 20px;
}

h4 {
  margin-bottom: 15px;
}

.control-card {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  transition: all 0.3s ease;
}

.control-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

.card-header {
  background: rgba(0, 0, 0, 0.2);
  padding: 12px 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.card-header h4 {
  margin: 0;
  color: white;
  font-size: 14px;
}

.card-body {
  padding: 16px;
}

/* 调整原有样式以适配 Card 布局 */
.control-group {
  margin-bottom: 12px;
}

.traffic-types {
  margin-top: 0;
}

.traffic-types li {
  margin-bottom: 8px;
}

.button-group {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

.button-group button {
  flex: 1;
  padding: 8px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.allow-btn {
  background-color: #4CAF50;
  color: white;
}

.block-btn {
  background-color: #F44336;
  color: white;
}

/* 自定义滚动条 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 255, 255, 0.05);
}

::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.2);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(255, 255, 255, 0.3);
}
</style>