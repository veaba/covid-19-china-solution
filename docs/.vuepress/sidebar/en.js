/***********************
 * @desc 该文件废弃，暂不用
 * @name JS
 * @author Jo.gel
 * @date 2020年3月27日17:13:05
 ***********************/
const {
	tfLinks,
} = require('./links');
module.exports = {
	'/en/manual/': [
		{
			title: 'Introduction',
			collapsable: false,
			link:'/en/manual/',
			children: tfLinks
		}
	]
};
