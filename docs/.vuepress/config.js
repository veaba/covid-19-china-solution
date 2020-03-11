module.exports = {
	base: "/covid-19-china-solution/",
	title: "COVID-19 中国方案",
	author: "veaba",
	description: "中国在应对COVID-19（新冠病毒）所采取多种举措被整理成为文档的中国版应对方案",
	displayAllHeaders: true, // 默认值：false
	// locales:{},//多语言支持 https://vuepress.vuejs.org/zh/guide/i18n.html
	scss: {},
	locales: {
		"/": {
			lang: "zh-CN",
			title: "COVID-19 中国方案",
			description: "中国在应对COVID-19（新冠病毒）所采取多种举措被整理成为文档的中国版应对方案"
		},
		"/en/": {
			lang: "en-US",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		}
	},
	themeConfig: {
		repo: "veaba/covid-19-china-solution",
		logo: "/favicon.png",
		locales: {
			//主站是中文版
			"/": {
				label: "简体中文",
				selectText: "选择语言",
				editLinkText: "在Github上编辑此页",
				nav: require("./nav/zh"),
				sidebar: require("./sidebar/zh")
			},
			//英文版
			"/en/": {
				label: "English",
				selectText: "Languages",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/en"),
				sidebar: require("./sidebar/en")
			}
		}
	},
	// 修改内部webpack的配置
	chinWebpack: (config, isServer) => {
	},
	// vuepress-plugin-container 容器
	plugins: [
		// tip
		[
			"container",
			{
				type: "tip",
				before: title =>
					`<div class="tip custom-block"> <p class="title">${title}</p>`,
				after: "</div>"
			}
		],
		[
			"container",
			{
				type: "warning",
				before: title =>
					`<div class="warning custom-block"> <p class="title">${title}</p>`,
				after: "</div>"
			}
		],
		[
			"container",
			{
				type: "danger",
				before: title =>
					`<div class="danger custom-block"> <p class="title">${title}</p>`,
				after: "</div>"
			}
		]
		// require('./vuepress-plugin-tensoflow')
	],
	extraWatchFiles: [".vuepress/nav/en.js", ".vuepress/nav/zh.js"]
};
