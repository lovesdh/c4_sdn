<template>
    <nav class="nav-top">
        <!-- 左侧项目名称模块 -->
        <div class="project-name">
            <span class="logo"><span class="logo-icon">🔷</span>微光卫士</span>
            <span class="project-title">网络监控系统</span>
        </div>

        <!-- 中间导航模块 -->
        <ul class="nav-ul">
            <li v-for="(item, index) in navItems" :key="item.name"
                class="nav-li"
                :class="{'active': $route.name === item.name}"
                @click="navigateTo(item)">
                <router-link :to="{ name: item.name}" class="page-link">
                    {{ item.label }}
                </router-link>
            </li>
        </ul>

        <!-- 右侧用户信息模块 -->
        <div class="user-info">
            <div class="user-actions">
                <div v-for="action in userActions" :key="action.name"
                     class="action-item"
                     @click="handleAction(action.name)">
                    <i :class="`icon-${action.name}`"></i>
                    <span>{{ action.label }}</span>
                </div>
            </div>
        </div>
    </nav>
</template>

<script>
export default {
    name: 'NavBar',
    data() {
        return {
            navItems: [
                { name: 'topology', label: '网络拓扑' },
                { name: 'deviceinfo', label: '设备信息' },
                { name: 'linkinfo', label: '链路信息' },
                { name: 'TrafficMonitor', label: '清洗信息' }
            ],
            userActions: [
                { name: 'tool', label: '制作团队' },
                { name: 'support', label: '支持' },
                { name: 'user', label: '账号' }
            ]
        }
    },
    methods: {
        navigateTo(item) {
            if (this.$route.name !== item.name) {
                this.$router.push({ name: item.name });
            }
        },
        handleAction(action) {
            // 处理用户操作
            console.log(`Action: ${action}`);
        }
    }
}
</script>

<style lang="scss" scoped>
.nav-top {
    width: 100%;
    height: 56px;
    background: linear-gradient(90deg, #0a1a2c 0%, #0f2b4a 100%);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    border-bottom: 1px solid rgba(74, 107, 255, 0.2);
    position: relative;
    z-index: 100;
}

/* 左侧项目名称 */
.project-name {
    display: flex;
    align-items: center;
    min-width: 200px;

    .logo {
        font-size: 20px;
        font-weight: bold;
        color: #fff;
        margin-right: 15px;
        text-shadow: 0 0 5px rgba(74, 107, 255, 0.5);
    }

    .project-title {
        font-size: 16px;
        color: rgba(255, 255, 255, 0.8);
    }
}

/* 中间导航 */
.nav-ul {
    display: flex;
    height: 100%;
    margin: 0;
    padding: 0;
    list-style: none;
}

.nav-li {
    height: 100%;
    display: flex;
    align-items: center;
    padding: 0 25px;
    position: relative;
    cursor: pointer;

    &:hover {
        background: rgba(74, 107, 255, 0.1);
    }

    &.active {
        background: rgba(74, 107, 255, 0.15);

        &::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, #4a6bff, #8a2be2);
            border-radius: 2px 2px 0 0;
            animation: fadeIn 0.2s ease-out;
        }
    }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(3px); }
    to { opacity: 1; transform: translateY(0); }
}

.page-link {
    font-size: 14px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.8);
    text-decoration: none;
    transition: color 0.15s ease-out;

    .active & {
        color: #fff;
        font-weight: 600;
    }
}

/* 右侧用户信息 */
.user-info {
    display: flex;
    align-items: center;
    min-width: 300px;
    justify-content: flex-end;
}

.user-actions {
    display: flex;
    align-items: center;
    position: relative;
    right: 50px;
    .action-item {
        display: flex;
        align-items: center;
        margin-left: 15px;
        padding: 5px 10px;
        border-radius: 4px;
        cursor: pointer;
        transition: background-color 0.15s ease-out;

        &:hover {
            background: rgba(74, 107, 255, 0.1);
        }

        i {
            width: 16px;
            height: 16px;
            margin-right: 5px;
            background: rgba(255, 255, 255, 0.7);
            mask-size: contain;
            mask-position: center;
            mask-repeat: no-repeat;
        }

        .icon-tool {
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M21.71 11.29l-9-9c-.39-.39-1.02-.39-1.41 0l-9 9c-.39.39-.39 1.02 0 1.41l9 9c.39.39 1.02.39 1.41 0l9-9c.39-.38.39-1.01 0-1.41zM14 14.5V12h-4v2.5l-4-4 4-4V10h4V6.5l4 4-4 4z'/%3E%3C/svg%3E");
        }

        .icon-support {
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z'/%3E%3C/svg%3E");
        }

        .icon-user {
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/%3E%3C/svg%3E");
        }

        span {
            font-size: 13px;
            color: rgba(255, 255, 255, 0.8);
        }
    }
}
</style>