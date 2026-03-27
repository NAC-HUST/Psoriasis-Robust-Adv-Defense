import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Psoriasis Robust Adv&Defense',
  description: '银屑病医学影像诊断的对抗攻防协同鲁棒性增强方法研究',
  lang: 'zh-CN',
  base: '/Psoriasis-Robust-Adv-Defense/',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: true,
  markdown: {
    math: true,
    lineNumbers: true
  },
  head: [
    ['link', { rel: 'preconnect', href: 'https://fonts.googleapis.com' }],
    ['link', { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' }],
    ['link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap' }]
  ],
  themeConfig: {
    siteTitle: 'PsoraDefense Docs',
    outline: {
      label: '本页导航',
      level: [2, 3]
    },
    search: {
      provider: 'local'
    },
    nav: [
      { text: '首页', link: '/' },
      { text: '入门', link: '/getting-started/overview' },
      { text: '文档', link: '/documentation/cli-reference' },
      { text: '研究', link: '/research/methodology' },
      { text: '贡献', link: '/contributing/guide' },
      { text: 'GitHub', link: 'https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense' }
    ],
    sidebar: [
      {
        text: '入门',
        items: [
          { text: '项目概览', link: '/getting-started/overview' },
          { text: '安装指南', link: '/getting-started/installation' },
          { text: '快速开始', link: '/getting-started/quickstart' }
        ]
      },
      {
        text: '文档',
        items: [
          { text: '项目结构', link: '/documentation/structure' },
          { text: '数据预处理', link: '/documentation/preprocessing' },
          { text: '模型训练', link: '/documentation/training' },
          { text: '对抗攻击', link: '/documentation/attack' },
          { text: 'CLI 参考', link: '/documentation/cli-reference' }
        ]
      },
      {
        text: '研究',
        items: [
          { text: '技术方案', link: '/research/methodology' },
          { text: '算法详解', link: '/research/algorithm' },
          { text: '实验结果', link: '/research/results' }
        ]
      },
      {
        text: '贡献',
        items: [
          { text: '贡献指南', link: '/contributing/guide' },
          { text: '代码规范', link: '/contributing/code-style' },
          { text: '开发流程', link: '/contributing/workflow' }
        ]
      },
      {
        text: '其他',
        items: [{ text: '许可证', link: '/license' }]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/NAC-HUST/Psoriasis-Robust-Adv-Defense' }
    ],
    footer: {
      message: 'Licensed under GPLv3',
      copyright: 'Copyright © 2024-2026 HUST Researchers'
    }
  }
})
