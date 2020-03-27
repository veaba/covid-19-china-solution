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
		},
		"/arab/": {
			lang: "arab",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/br/": {
			lang: "br",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/farsi/": {
			lang: "farsi",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/es/": {
			lang: "es",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/fr/": {
			lang: "fr",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/german/": {
			lang: "german",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/id/": {
			lang: "id",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/italy/": {
			lang: "italy",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/jp/": {
			lang: "jp",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/kr/": {
			lang: "kr",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/ru/": {
			lang: "ru",
			title: "A COVID-19 china solution",
			description: "Vue 驱动的静态网站生成器"
		},
		"/vi/": {
			lang: "vi",
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
			},
			// 阿拉伯 
			"/arab/": {
				label: "Arabic",
				selectText: "Arabic",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/arab"),
				sidebar: require("./sidebar/arab")
			},
			// 葡萄牙语
			"/br/": {
				label: "Português",
				selectText: "Português",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/br"),
				sidebar: require("./sidebar/br")
			},
			// 波斯语
			"/farsi/": {
				label: "Farsi",
				selectText: "Farsi",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/farsi"),
				sidebar: require("./sidebar/farsi")
			},
			// 法语
			"/fr/": {
				label: "Français",
				selectText: "Français",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/fr"),
				sidebar: require("./sidebar/fr")
			},
			// 西班牙语
			"/es/": {
				label: "español",
				selectText: "español",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/es"),
				sidebar: require("./sidebar/es")
			},
			// 德语
			"/german/": {
				label: "Deutsch",
				selectText: "Deutsch",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/german"),
				sidebar: require("./sidebar/german")
			},
			// 印度尼西亚
			"/id/": {
				label: "Bahasa Indonesia",
				selectText: "Bahasa Indonesia",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/id"),
				sidebar: require("./sidebar/id")
			},
			// 意大利语
			"/italy": {
				label: "Italiano",
				selectText: "Italiano",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/italy"),
				sidebar: require("./sidebar/italy")
			},
			// 日语
			"/jp/": {
				label: "日本語",
				selectText: "日本語",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/jp"),
				sidebar: require("./sidebar/jp")
			},
			// 韩语
			"/kr/": {
				label: "한국어",
				selectText: "한국어",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/kr"),
				sidebar: require("./sidebar/kr")
			},
			// 俄语
			"/ru/": {
				label: "русский язык ",
				selectText: "русский язык ",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/ru"),
				sidebar: require("./sidebar/ru")
			},
			// 越南语
			"/vi/": {
				label: "Tiếng Việt",
				selectText: "Tiếng Việt",
				editLinkText: "Edit this page on Github",
				nav: require("./nav/vi"),
				sidebar: require("./sidebar/vi")
			},
		}
	},
	// 修改内部webpack的配置
	chinWebpack: (config, isServer) => {},
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