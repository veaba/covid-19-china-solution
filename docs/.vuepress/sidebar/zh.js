/***********************
 * @name JS
 * @author Jo.gel
 * @date 2019/8/2 0002
 ***********************/

const {
	tfLinks,
} = require('./links');
module.exports = {
	'/': [
		{'title': 'tf', 'collapsabel': false, 'children': tfLinks},
	]
};
