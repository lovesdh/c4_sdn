<template>
  <div class="network-monitor">
    <!-- 攻击模拟按钮 -->
    <button class="attack-button" @click="toggleAttack" style="opacity: 0; cursor: help;">
      {{ isUnderAttack ? '解除攻击' : '模拟攻击' }}
    </button>

    <!-- 左侧链路列表 -->
    <div class="link-list" style="overflow-y: scroll;">
      <h3 style="color: white;">链路列表</h3>
      <ul>
        <li
          v-for="link in links"
          :key="link.id"
          :class="['link-item', link.status]"
          @click="selectLink(link)"
        >
          <span class="link-name">{{ link.name }}</span>
          <span class="link-status">{{ link.statusText }}</span>
          <span v-if="link.usage" class="link-usage">{{ link.usage }}%</span>
        </li>
      </ul>
    </div>

    <!-- 右侧流量监控图 -->
    <div class="traffic-monitor">
      <h3 style="color: white;">流量监控</h3>
      <div class="chart-container">
        <canvas ref="trafficChart"></canvas>
      </div>
      <div class="time-labels">
        <span v-for="(time, index) in timeLabels" :key="index">{{ time }}</span>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, watch } from 'vue';
import Chart from 'chart.js/auto';

export default {
  name: 'NetworkMonitor',
  setup() {
    // 链路数据（预设流量模式）
    const links = ref([
      {
        id: 1,
        name: 'leaf1-h1a',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [20, 22, 19, 21, 23, 20, 22, 21, 20, 19],
        attackPattern: [85, 88, 90, 92, 94, 95, 96, 94, 92, 90],
        recoveryPattern: [90, 85, 80, 75, 70, 65, 60, 55, 50, 45]
      },
      {
        id: 2,
        name: 's1-leaf1',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [15, 16, 14, 15, 17, 16, 15, 14, 15, 16],
        attackPattern: [80, 82, 85, 83, 84, 86, 85, 84, 83, 82],
        recoveryPattern: [82, 75, 70, 65, 60, 55, 50, 45, 40, 35]
      },
      {
        id: 3,
        name: 'leaf2-h2a',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [25, 26, 24, 25, 27, 26, 25, 24, 25, 26],
        attackPattern: [25, 26, 24, 25, 27, 26, 25, 24, 25, 26],
        recoveryPattern: [25, 26, 24, 25, 27, 26, 25, 24, 25, 26]
      },
      {
        id: 4,
        name: 'leaf2-h2b',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [18, 19, 17, 18, 20, 19, 18, 17, 18, 19],
        attackPattern: [18, 19, 17, 18, 20, 19, 18, 17, 18, 19],
        recoveryPattern: [18, 19, 17, 18, 20, 19, 18, 17, 18, 19]
      },
      {
        id: 5,
        name: 'leaf3-h3a',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [12, 13, 11, 12, 14, 13, 12, 11, 12, 13],
        attackPattern: [12, 13, 11, 12, 14, 13, 12, 11, 12, 13],
        recoveryPattern: [12, 13, 11, 12, 14, 13, 12, 11, 12, 13]
      },
      {
        id: 6,
        name: 'leaf3-h3b',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [22, 23, 21, 22, 24, 23, 22, 21, 22, 23],
        attackPattern: [22, 23, 21, 22, 24, 23, 22, 21, 22, 23],
        recoveryPattern: [22, 23, 21, 22, 24, 23, 22, 21, 22, 23]
      },
      {
        id: 7,
        name: 'leaf4-h4a',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [30, 31, 29, 30, 32, 31, 30, 29, 30, 31],
        attackPattern: [30, 31, 29, 30, 32, 31, 30, 29, 30, 31],
        recoveryPattern: [30, 31, 29, 30, 32, 31, 30, 29, 30, 31]
      },
      {
        id: 8,
        name: 'leaf4-h4b',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [28, 29, 27, 28, 30, 29, 28, 27, 28, 29],
        attackPattern: [28, 29, 27, 28, 30, 29, 28, 27, 28, 29],
        recoveryPattern: [28, 29, 27, 28, 30, 29, 28, 27, 28, 29]
      },
      {
        id: 9,
        name: 'leaf5-h5a',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [35, 36, 34, 35, 37, 36, 35, 34, 35, 36],
        attackPattern: [35, 36, 34, 35, 37, 36, 35, 34, 35, 36],
        recoveryPattern: [35, 36, 34, 35, 37, 36, 35, 34, 35, 36]
      },
      {
        id: 10,
        name: 'leaf5-h5b',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [40, 41, 39, 40, 42, 41, 40, 39, 40, 41],
        attackPattern: [40, 41, 39, 40, 42, 41, 40, 39, 40, 41],
        recoveryPattern: [40, 41, 39, 40, 42, 41, 40, 39, 40, 41]
      },
      {
        id: 11,
        name: 'leaf5-h5c',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [38, 39, 37, 38, 40, 39, 38, 37, 38, 39],
        attackPattern: [38, 39, 37, 38, 40, 39, 38, 37, 38, 39],
        recoveryPattern: [38, 39, 37, 38, 40, 39, 38, 37, 38, 39]
      },
      {
        id: 12,
        name: 'spine1-leaf1',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [45, 46, 44, 45, 47, 46, 45, 44, 45, 46],
        attackPattern: [90, 92, 94, 95, 96, 97, 98, 97, 96, 95],
        recoveryPattern: [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
      },
      {
        id: 13,
        name: 'spine1-leaf2',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [42, 43, 41, 42, 44, 43, 42, 41, 42, 43],
        attackPattern: [42, 43, 41, 42, 44, 43, 42, 41, 42, 43],
        recoveryPattern: [42, 43, 41, 42, 44, 43, 42, 41, 42, 43]
      },
      {
        id: 14,
        name: 'spine1-leaf3',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [38, 39, 37, 38, 40, 39, 38, 37, 38, 39],
        attackPattern: [38, 39, 37, 38, 40, 39, 38, 37, 38, 39],
        recoveryPattern: [38, 39, 37, 38, 40, 39, 38, 37, 38, 39]
      },
      {
        id: 15,
        name: 'spine1-leaf4',
        status: 'normal',
        statusText: '正常',
        usage: null,
        normalPattern: [35, 36, 34, 35, 37, 36, 35, 34, 35, 36],
        attackPattern: [35, 36, 34, 35, 37, 36, 35, 34, 35, 36],
        recoveryPattern: [35, 36, 34, 35, 37, 36, 35, 34, 35, 36]
      }
    ]);

    // 当前选中的链路
    const selectedLink = ref(links.value[0]);
    const isUnderAttack = ref(false);
    const isRecovering = ref(false);

    // 时间标签
    const timeLabels = ref([
      '10:57:07', '10:57:07', '10:57:07', '10:57:07',
      '10:57:12', '10:57:12', '10:57:13', '10:57:14',
      '10:57:15', '10:57:15'
    ]);

    // 图表引用
    const trafficChart = ref(null);
    let chartInstance = null;

    // 生成链路流量数据
    const generateTrafficData = () => {
      if (isRecovering.value) {
        return selectedLink.value.recoveryPattern;
      } else if (isUnderAttack.value) {
        return selectedLink.value.attackPattern;
      } else {
        return selectedLink.value.normalPattern;
      }
    };

    // 初始化图表
    const initChart = () => {
      if (chartInstance) {
        chartInstance.destroy();
      }

      const ctx = trafficChart.value.getContext('2d');
      const trafficData = generateTrafficData();

      chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: Array(10).fill(''),
          datasets: [{
            label: '流量',
            data: trafficData,
            borderColor: isUnderAttack.value ? '#ff5555' : '#64ffda',
            backgroundColor: isUnderAttack.value ? 'rgba(255, 85, 85, 0.1)' : 'rgba(100, 255, 218, 0.1)',
            borderWidth: 2,
            tension: 0.1,
            fill: true
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              grid: {
                color: 'rgba(255, 255, 255, 0.1)'
              },
              ticks: {
                color: '#e6f1ff',
                callback: function(value) {
                  return value + '%';
                }
              }
            },
            x: {
              grid: {
                display: false
              },
              ticks: {
                display: false
              }
            }
          },
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  return '流量: ' + context.parsed.y + '%';
                }
              }
            }
          },
          elements: {
            point: {
              radius: 0
            }
          }
        }
      });
    };

    // 选择链路
    const selectLink = (link) => {
      selectedLink.value = link;
    };

    // 切换攻击状态
    const toggleAttack = () => {
      if (isUnderAttack.value) {
        // 从攻击状态切换到恢复状态
        isRecovering.value = true;
        isUnderAttack.value = false;

        // 更新相关链路状态
        updateLinkStatus(false);
      } else {
        // 从正常状态切换到攻击状态
        isUnderAttack.value = true;
        isRecovering.value = false;

        // 更新相关链路状态
        updateLinkStatus(true);
      }

      // 重新渲染图表
      initChart();
    };

    // 更新链路状态
    const updateLinkStatus = (isAttack) => {
      links.value.forEach(link => {
        // 更新与h1相关的链路状态
        if (link.name.includes('h1') || link.name === 'spine1-leaf1') {
          if (isAttack) {
            link.status = 'congested';
            link.statusText = '拥塞';
          } else {
            link.status = 'normal';
            link.statusText = '正常';
          }
        }
      });
    };

    // 监听选中的链路变化
    watch(selectedLink, () => {
      if (chartInstance) {
        initChart();
      }
    });

    // 组件挂载时初始化图表
    onMounted(() => {
      initChart();
    });

    return {
      links,
      timeLabels,
      trafficChart,
      selectLink,
      toggleAttack,
      isUnderAttack
    };
  }
};
</script>

<style scoped>
.network-monitor {
  display: flex;
  height: 100vh;
  background-color: #0a192f;
  color: #e6f1ff;
  font-family: Arial, sans-serif;
  position: relative;
}

.attack-button {
  position: absolute;
  top: 20px;
  right: 20px;
  padding: 10px 20px;
  background-color: #ff5555;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  z-index: 100;
  font-weight: bold;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
  transition: background-color 0.3s;
}

.attack-button:hover {
  background-color: #ff3333;
}

.link-list::-webkit-scrollbar {
  display: none;
}

.link-list {
  width: 250px;
  padding: 20px;
  background-color: #112240;
  border-right: 1px solid #1e2a4a;
  overflow-y: auto;
}

.traffic-monitor {
  flex: 1;
  padding: 20px;
  display: flex;
  flex-direction: column;
}

h3 {
  color: #64ffda;
  margin-top: 0;
  margin-bottom: 20px;
}

.link-list ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.link-item {
  display: flex;
  flex-direction: column;
  padding: 12px;
  margin-bottom: 8px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.link-item:hover {
  background-color: rgba(30, 42, 74, 0.5);
}

.link-name {
  font-weight: bold;
  margin-bottom: 4px;
}

.link-status {
  font-size: 14px;
  margin-bottom: 4px;
}

.link-usage {
  font-size: 12px;
  opacity: 0.8;
}

/* 不同状态的颜色 */
.link-item.congested {
  border-left: 3px solid #FFC107;
}

.link-item.idle {
  border-left: 3px solid #4CAF50;
}

.link-item.normal {
  border-left: 3px solid #2196F3;
}

.link-item.clone {
  border-left: 3px solid #9C27B0;
}

.chart-container {
  flex: 1;
  position: relative;
  margin-bottom: 20px;
}

.time-labels {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: rgba(230, 241, 255, 0.7);
}
</style>