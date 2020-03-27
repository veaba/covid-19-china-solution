const {
	tfLinks,
} = require('./links');
module.exports = {
	'/': [
		{'title': 'tf', 'collapsabel': false, 'children': tfLinks},
	]
};
