/***********************
 * @desc 该文件废弃，暂不用
 * @name JS
 * @author Jo.gel
 * @date 2019/8/2 0002
 ***********************/
const {
	introductionLinks,
	installationLinks,
	theMongoShellLinks,
	mongoDBCRUDOperationsLinks,
	aggregationLinks,
	dataModelsLinks,
	transactionsLinks,
	indexesLinks,
	securityLinks,
	changeStreamsLinks,
	replicationLinks,
	shardingLinks,
	administrationLinks,
	storageLinks,
	frequentlyAskedQuestionsLinks,
	referenceLinks,
	releaseNotesLinks,
	technicalSupportLinks
} = require('./links');
module.exports = {
	'/en/manual/': [
		{
			title: 'Introduction',
			collapsable: false,
			link:'/en/manual/',
			children: introductionLinks
		},
		{
			title: 'Installation',
			collapsable: false,
			link:"/en/manual/installation/",
			children: installationLinks
		},
		{
			title: 'The mongo Shell',
			collapsable: false,
			children: theMongoShellLinks
		},
		{
			title: 'MongoDB CRUD Operations',
			collapsable: false,
			children: mongoDBCRUDOperationsLinks
		},
		{
			title: 'Aggregation',
			collapsable: false,
			children: aggregationLinks
		},
		{
			title: 'Data Models',
			collapsable: false,
			children: dataModelsLinks
		},
		{
			title: 'Transactions',
			collapsable: false,
			children: transactionsLinks
		},
		{
			title: 'Indexes',
			collapsable: false,
			children: indexesLinks
		},
		{
			title: 'Security',
			collapsable: false,
			children: securityLinks
		},
		{
			title: 'Change Streams',
			collapsable: false,
			children: changeStreamsLinks
		},
		{
			title: 'Replication',
			collapsable: false,
			children: replicationLinks
		},
		{
			title: 'Sharding',
			collapsable: false,
			children: shardingLinks
		},
		{
			title: 'Administration',
			collapsable: false,
			children: administrationLinks
		},
		{
			title: 'Storage',
			collapsable: false,
			children: storageLinks
		},
		{
			title: 'Frequently Asked Questions',
			collapsable: false,
			children: frequentlyAskedQuestionsLinks
		},
		{
			title: 'Reference',
			collapsable: false,
			children: referenceLinks
		},
		{
			title: 'Release Notes',
			collapsable: false,
			children: releaseNotesLinks
		},
		'support',
		// {
		// 	title: 'Technical Support',
		// 	collapsable: false,
		// 	children: technicalSupportLinks
		// }
	
	]
};
