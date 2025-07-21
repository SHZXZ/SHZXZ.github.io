# lylzxz的个人博客

基于 [Astro Gyoza](https://github.com/lxchapu/astro-gyoza) 模板构建的个人博客网站，采用现代化的技术栈打造优雅的阅读体验。

![Astro Version](https://img.shields.io/badge/astro-4.6-red)
![Node Version](https://img.shields.io/badge/node-18.18-green)
![License](https://img.shields.io/badge/license-MIT-blue)

🌐 **在线访问**: [https://SHZXZ.github.io](https://SHZXZ.github.io)

## ✨ 特性亮点

### 🎨 用户体验

- ✅ **响应式设计** - 完美适配桌面端和移动端
- ✅ **夜间模式** - 自动/手动主题切换，护眼舒适
- ✅ **动态主题色** - 丰富的颜色主题随机切换
- ✅ **流畅动画** - 基于 Framer Motion 的精美过渡效果
- ✅ **特殊日期变灰** - 哀悼日等特殊日期自动启用灰色模式

### 📝 内容管理

- ✅ **Markdown 支持** - 完整的 Markdown 语法和扩展支持
- ✅ **数学公式** - 支持 KaTeX 数学公式渲染
- ✅ **代码高亮** - 基于 Shiki 的精美代码高亮
- ✅ **阅读时间** - 自动计算文章阅读时间
- ✅ **文章置顶** - 支持重要文章置顶功能
- ✅ **草稿模式** - 支持草稿文章的编写和管理

### 🔍 导航与搜索

- ✅ **智能分类** - 自动分类管理和标签系统
- ✅ **时间轴归档** - 优雅的时间轴展示历史文章
- ✅ **全站搜索** - 基于 Pagefind 的快速全文搜索
- ✅ **目录导航** - 自动生成文章目录和阅读进度

### 🌍 SEO 和集成

- ✅ **SEO 优化** - 完整的 meta 标签和 OpenGraph 支持
- ✅ **站点地图** - 自动生成 XML 站点地图
- ✅ **RSS 订阅** - 支持 RSS 2.0 订阅
- ✅ **评论系统** - 集成 Waline 评论系统
- ✅ **网站统计** - 支持多种分析工具（Google Analytics、Umami 等）

### 🛠️ 开发体验

- ✅ **TypeScript** - 完整的类型安全支持
- ✅ **自动化脚本** - 便捷的文章/友链/项目创建脚本
- ✅ **代码规范** - ESLint + Prettier 代码质量保证
- ✅ **Git Hooks** - 自动化代码检查和格式化

## 🚀 技术栈

### 核心框架

- **[Astro](https://astro.build/)** - 现代化静态站点生成器
- **[React](https://reactjs.org/)** - 用户界面库
- **[TypeScript](https://www.typescriptlang.org/)** - 类型安全的 JavaScript

### 样式和动画

- **[Tailwind CSS](https://tailwindcss.com/)** - 原子化 CSS 框架
- **[Framer Motion](https://www.framer.com/motion/)** - 流畅的动画库

### 状态管理和工具

- **[Jotai](https://jotai.org/)** - 原子化状态管理
- **[Swup](https://swup.js.org/)** - 页面切换动画
- **[Pagefind](https://pagefind.app/)** - 静态站点搜索

### Markdown 处理

- **[Shiki](https://shiki.matsu.io/)** - 代码语法高亮
- **[KaTeX](https://katex.org/)** - 数学公式渲染
- **[remark](https://remark.js.org/)** & **[rehype](https://github.com/rehypejs/rehype)** - Markdown 处理管道

## 📁 项目结构

```
SHZXZ.github.io/
├── 📁 public/                           # 静态资源目录
│   ├── 📁 fonts/                       # 字体文件
│   │   ├── atkinson-bold.woff          # Atkinson 粗体字体
│   │   ├── atkinson-regular.woff       # Atkinson 常规字体
│   │   ├── iconfont.ttf                # 图标字体 TTF
│   │   ├── iconfont.woff               # 图标字体 WOFF
│   │   └── iconfont.woff2              # 图标字体 WOFF2
│   ├── favicon.ico                     # 网站图标
│   └── apple-touch-icon.png            # 苹果设备图标
│
├── 📁 scripts/                          # 自动化脚本目录
│   ├── new-post.js                     # 创建新文章脚本
│   ├── new-friend.js                   # 创建新友链脚本
│   ├── new-project.js                  # 创建新项目脚本
│   └── utils.js                        # 脚本工具函数
│
├── 📁 src/                             # 源代码目录
│   ├── 📁 assets/                      # 项目资源
│   │   └── signature.svg               # 个人签名图标
│   │
│   ├── 📁 components/                  # React/Astro 组件库
│   │   ├── 📁 comment/                 # 评论系统组件
│   │   │   ├── Comments.astro          # 评论容器组件
│   │   │   ├── index.ts                # 评论组件导出
│   │   │   └── Waline.tsx              # Waline 评论组件
│   │   ├── 📁 footer/                  # 页脚组件
│   │   │   ├── Footer.astro            # 页脚主组件
│   │   │   ├── Link.astro              # 页脚链接组件
│   │   │   ├── RunningDays.tsx         # 运行天数组件
│   │   │   └── ThemeSwitch.tsx         # 主题切换组件
│   │   ├── 📁 head/                    # 页面头部组件
│   │   │   ├── AccentColorInjector.astro # 主题色注入器
│   │   │   ├── CommonHead.astro        # 通用头部信息
│   │   │   ├── index.ts                # 头部组件导出
│   │   │   ├── PrintVersion.astro      # 版本信息打印
│   │   │   ├── ThemeLoader.astro       # 主题加载器
│   │   │   └── WebAnalytics.tsx        # 网站分析组件
│   │   ├── 📁 head-gradient/           # 头部渐变效果
│   │   │   ├── HeadGradient.tsx        # 渐变背景组件
│   │   │   └── index.ts                # 渐变组件导出
│   │   ├── 📁 header/                  # 页面头部导航
│   │   │   ├── AnimatedLogo.tsx        # 动画 Logo
│   │   │   ├── BluredBackground.tsx    # 模糊背景效果
│   │   │   ├── Header.tsx              # 头部主组件
│   │   │   ├── HeaderContent.tsx       # 头部内容
│   │   │   ├── HeaderDrawer.tsx        # 移动端抽屉菜单
│   │   │   ├── HeaderMeta.tsx          # 头部元信息
│   │   │   ├── hooks.ts                # 头部相关 Hooks
│   │   │   └── SearchButton.tsx        # 搜索按钮
│   │   ├── 📁 hero/                    # 首页横幅区域
│   │   │   ├── Hero.astro              # 横幅主组件
│   │   │   └── SocialList.tsx          # 社交媒体列表
│   │   ├── 📁 post/                    # 文章相关组件
│   │   │   ├── ActionAside.tsx         # 文章侧边操作
│   │   │   ├── Outdate.tsx             # 过期提醒组件
│   │   │   ├── PostArchiveInfo.astro   # 文章归档信息
│   │   │   ├── PostCard.astro          # 文章卡片
│   │   │   ├── PostCardHoverOverlay.tsx # 文章卡片悬停效果
│   │   │   ├── PostCopyright.tsx       # 文章版权信息
│   │   │   ├── PostList.astro          # 文章列表
│   │   │   ├── PostMetaInfo.astro      # 文章元信息
│   │   │   ├── PostNav.astro           # 文章导航
│   │   │   ├── PostPagination.astro    # 文章分页
│   │   │   ├── PostToc.tsx             # 文章目录
│   │   │   ├── ReadingProgress.tsx     # 阅读进度
│   │   │   └── RelativeDate.tsx        # 相对时间显示
│   │   ├── 📁 provider/                # 状态提供者组件
│   │   │   └── Provider.tsx            # 全局状态提供者
│   │   ├── 📁 ui/                      # 基础 UI 组件
│   │   │   └── 📁 modal/               # 模态框组件
│   │   │       └── ModalStack.tsx      # 模态框堆栈管理
│   │   ├── AnimatedSignature.tsx       # 动画签名组件
│   │   ├── BackToTopFAB.tsx           # 返回顶部按钮
│   │   ├── CategoryList.astro          # 分类列表组件
│   │   ├── Flashlight.tsx              # 手电筒效果组件
│   │   ├── FriendList.astro            # 友链列表组件
│   │   ├── Highlight.astro             # 高亮文本组件
│   │   ├── MarkdownWrapper.astro       # Markdown 包装器
│   │   ├── ProjectList.astro           # 项目列表组件
│   │   ├── RootPortal.tsx              # 根门户组件
│   │   ├── SectionBlock.astro          # 区块组件
│   │   ├── TagList.astro               # 标签列表组件
│   │   ├── Timeline.astro              # 时间轴组件
│   │   ├── TimelineProgress.tsx        # 时间轴进度
│   │   └── ToastContainer.tsx          # 消息提示容器
│   │
│   ├── 📁 content/                     # Astro 内容集合
│   │   ├── config.ts                   # 内容类型定义和验证
│   │   ├── 📁 friends/                 # 友情链接数据
│   │   ├── 📁 posts/                   # 博客文章 Markdown 文件
│   │   ├── 📁 projects/                # 项目展示数据
│   │   └── 📁 spec/                    # 特殊页面内容
│   │
│   ├── 📁 hooks/                       # React Hooks
│   │   └── useDebounceValue.ts         # 防抖值 Hook
│   │
│   ├── 📁 layouts/                     # 页面布局组件
│   │   ├── Layout.astro                # 基础页面布局
│   │   ├── MarkdownLayout.astro        # Markdown 页面布局
│   │   └── PageLayout.astro            # 普通页面布局
│   │
│   ├── 📁 pages/                       # Astro 页面路由
│   │   ├── 📁 categories/              # 分类相关页面
│   │   │   ├── index.astro             # 分类列表页
│   │   │   └── [category].astro        # 单个分类页
│   │   ├── 📁 posts/                   # 文章相关页面
│   │   │   └── [...slug].astro         # 动态文章页面
│   │   ├── 📁 tags/                    # 标签相关页面
│   │   │   ├── index.astro             # 标签列表页
│   │   │   └── [tag].astro             # 单个标签页
│   │   ├── [...page].astro             # 首页分页路由
│   │   ├── [spec].astro                # 特殊页面路由
│   │   ├── 404.astro                   # 404 错误页面
│   │   ├── archives.astro              # 文章归档页面
│   │   ├── robots.txt.ts               # Robots.txt 生成
│   │   └── rss.xml.ts                  # RSS 订阅生成
│   │
│   ├── 📁 plugins/                     # Markdown 处理插件
│   │   ├── remarkEmbed.js              # 嵌入内容插件
│   │   ├── remarkReadingTime.js        # 阅读时间计算插件
│   │   ├── remarkSpoiler.js            # 剧透隐藏插件
│   │   ├── rehypeCodeBlock.js          # 代码块处理插件
│   │   ├── rehypeCodeHighlight.js      # 代码高亮插件
│   │   ├── rehypeHeading.js            # 标题处理插件
│   │   ├── rehypeImage.js              # 图片处理插件
│   │   ├── rehypeLink.js               # 链接处理插件
│   │   └── rehypeTableBlock.js         # 表格处理插件
│   │
│   ├── 📁 store/                       # Jotai 状态管理
│   │   ├── metaInfo.ts                 # 页面元信息状态
│   │   ├── modalStack.ts               # 模态框堆栈状态
│   │   ├── scrollInfo.ts               # 滚动信息状态
│   │   ├── theme.ts                    # 主题状态
│   │   └── viewport.ts                 # 视口信息状态
│   │
│   ├── 📁 styles/                      # 样式文件
│   │   ├── global.css                  # 全局样式
│   │   ├── iconfont.css                # 图标字体样式
│   │   ├── markdown.css                # Markdown 内容样式
│   │   ├── shiki.css                   # 代码高亮样式
│   │   ├── signature.css               # 签名动画样式
│   │   └── swup.css                    # 页面切换动画样式
│   │
│   ├── 📁 utils/                       # 工具函数
│   │   ├── content.ts                  # 内容处理工具
│   │   ├── date.ts                     # 日期处理工具
│   │   └── theme.ts                    # 主题处理工具
│   │
│   ├── config.json                     # 站点配置文件
│   └── env.d.ts                        # TypeScript 环境声明
│
├── astro.config.js                     # Astro 框架配置
├── commitlint.config.js                # 提交信息规范配置
├── tailwind.config.ts                  # Tailwind CSS 配置
├── tsconfig.json                       # TypeScript 配置
├── package.json                        # 项目依赖和脚本
├── pnpm-lock.yaml                      # pnpm 锁定文件
├── LICENSE                             # 开源许可证
└── README.md                           # 项目说明文档
```

## 🛠️ 快速开始

### 环境要求

- Node.js 18.18+
- pnpm (推荐) 或 npm

### 安装依赖

```bash
# 使用 pnpm (推荐)
pnpm install

# 或使用 npm
npm install
```

### 开发命令

```bash
# 启动开发服务器
pnpm dev                    # 访问 http://localhost:4321

# 构建生产版本
pnpm build                  # 输出到 ./dist/

# 预览构建结果
pnpm preview                # 预览生产构建

# 代码格式化
pnpm lint                   # 格式化所有代码

# 创建新内容
pnpm new-post              # 创建新文章
pnpm new-friend            # 创建新友链
pnpm new-project           # 创建新项目
```

## ⚙️ 配置说明

### 站点配置 (`src/config.json`)

```json
{
  "site": {
    "url": "你的网站URL",
    "title": "网站标题",
    "description": "网站描述",
    "keywords": "关键词",
    "lang": "zh-CN"
  },
  "author": {
    "name": "作者名称",
    "avatar": "头像链接",
    "twitterId": "@twitter用户名"
  },
  "hero": {
    "name": "显示名称",
    "bio": "个人简介",
    "socials": [
      {
        "name": "GitHub",
        "icon": "icon-github",
        "url": "https://github.com/你的用户名",
        "color": "rgb(24, 23, 23)"
      }
    ]
  }
}
```

### 主要功能配置

- **评论系统**: 配置 Waline 评论服务器 URL
- **网站统计**: 支持 Google Analytics、Umami、Microsoft Clarity
- **主题色彩**: 自定义主题色彩配置
- **导航菜单**: 自定义顶部导航菜单

## 📝 内容创建

### 创建新文章

```bash
pnpm new-post
```

文章模板：

```markdown
---
title: 文章标题
date: 2024-01-01T00:00:00.000Z
summary: 文章摘要
category: 分类名称
tags: [标签1, 标签2]
cover: 封面图片URL (可选)
sticky: 0 # 置顶优先级，数字越大越靠前
comments: true # 是否开启评论
draft: false # 是否为草稿
---

这里是文章内容...
```

### 创建友情链接

```bash
pnpm new-friend
```

### 创建项目展示

```bash
pnpm new-project
```

## 🎨 自定义样式

### 主题色彩

在 `src/config.json` 中配置主题色彩：

```json
{
  "color": {
    "accent": [
      { "light": "#F55555", "dark": "#FCCF31" },
      { "light": "#0396FF", "dark": "#ABDCFF" }
    ]
  }
}
```

### 自定义样式

- `src/styles/global.css` - 全局样式
- `src/styles/markdown.css` - Markdown 内容样式
- `src/styles/shiki.css` - 代码高亮样式

## 📱 部署指南

### GitHub Pages

1. 在 GitHub 上创建仓库
2. 推送代码到 `main` 分支
3. 在仓库设置中启用 GitHub Pages
4. 选择 GitHub Actions 作为部署源

### Vercel/Netlify

1. 连接 GitHub 仓库
2. 设置构建命令：`pnpm build`
3. 设置输出目录：`dist`
4. 部署完成

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 创建 Pull Request

## 📄 许可证

本项目基于 MIT 许可证开源 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 [lxchapu](https://github.com/lxchapu) 开源的优秀模板 [astro-gyoza](https://github.com/lxchapu/astro-gyoza)
- 感谢 [Astro](https://astro.build/) 团队提供的优秀框架
- 感谢所有开源项目的贡献者们

## 💬 联系方式

- **Email**: 1454285348@qq.com
- **GitHub**: [@SHZXZ](https://github.com/SHZXZ)
- **X (Twitter)**: [@SH9277390497666](https://x.com/SH9277390497666)

---

<div align="center">

**[⬆ 回到顶部](#lylzxz的个人博客)**

Made with ❤️ by [lylzxz](https://github.com/SHZXZ)

</div>
