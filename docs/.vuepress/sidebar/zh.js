/***********************
 * @name JS
 * @author Jo.gel
 * @date 2019/8/2 0002
 ***********************/

const {
	tfLinks,
	tfAudioLinks,
	tfAutographLinks,
	tfBitwiseLinks,
	tfCompatLinks,
	tfConfigLinks,
	tfDataLinks,
	tfDebuggingLinks,
	tfDistributeLinks,
	tfDtypesLinks,
	tfEstimatorLinks,
	tfErrorsLinks,
	tfExperimentalLinks,
	tfFeatureColumnLinks,
	tfGraphUtilLinks,
	tfImageLinks,
	tfInitializersLinks,
	tfIOLinks,
	tfLinalgLinks,
	tfKerasLinks,
	tfLiteLinks,
	tfLookupLinks,
	tfLossesLinks,
	tfMathLinks,
	tfMetricsLinks,
	tfNestLinks,
	tfNNLinks,
	tfOptimizersLinks,
	tfQuantizationLinks,
	tfQueueLinks,
	tfRaggedLinks,
	tfRandomLinks,
	tfRawOpsLinks,
	tfSetsLinks,
	tfSignalLinks,
	tfSparseLinks,
	tfStringsLinks,
	tfSummaryLinks,
	tfSysconfigLinks,
	tfTestLinks,
	tfTpuLinks,
	tfTrainLinks,
	tfVersionLinks,
	tfXlaLinks,
	allSymbolsLinks
} = require('./links');
module.exports = {
	'/': [
		{'title': 'tf', 'collapsabel': false, 'children': tfLinks},
		{'title': 'tf.audio', 'collapsabel': false, 'children': tfAudioLinks},
		{'title': 'tf.autograph', 'collapsabel': false, 'children': tfAutographLinks},
		{'title': 'tf.bitwise', 'collapsabel': false, 'children': tfBitwiseLinks},
		{'title': 'tf.compat', 'collapsabel': false, 'children': tfCompatLinks},
		{'title': 'tf.config', 'collapsabel': false, 'children': tfConfigLinks},
		{'title': 'tf.data', 'collapsabel': false, 'children': tfDataLinks},
		{'title': 'tf.debugging', 'collapsabel': false, 'children': tfDebuggingLinks},
		{'title': 'tf.distribute', 'collapsabel': false, 'children': tfDistributeLinks},
		{'title': 'tf.dtypes', 'collapsabel': false, 'children': tfDtypesLinks},
		{'title': 'tf.errors', 'collapsabel': false, 'children': tfErrorsLinks},
		{'title': 'tf.estimator', 'collapsabel': false, 'children': tfEstimatorLinks},
		{'title': 'tf.experimental', 'collapsabel': false, 'children': tfExperimentalLinks,'isExperiment': true},
		{'title': 'tf.feature_column', 'collapsabel': false, 'children': tfFeatureColumnLinks},
		{'title': 'tf.graph_util', 'collapsabel': false, 'children': tfGraphUtilLinks},
		{'title': 'tf.image', 'collapsabel': false, 'children': tfImageLinks},
		{'title': 'tf.initializers', 'collapsabel': false, 'children': tfInitializersLinks},
		{'title': 'tf.io', 'collapsabel': false, 'children': tfIOLinks},
		{'title': 'tf.keras', 'collapsabel': false, 'children': tfKerasLinks},
		{'title': 'tf.linalg', 'collapsabel': false, 'children': tfLinalgLinks},
		{'title': 'tf.lite', 'collapsabel': false, 'children': tfLiteLinks},
		{'title': 'tf.lookup', 'collapsabel': false, 'children': tfLookupLinks},
		{'title': 'tf.losses', 'collapsabel': false, 'children': tfLossesLinks},
		{'title': 'tf.math', 'collapsabel': false, 'children': tfMathLinks},
		{'title': 'tf.metrics', 'collapsabel': false, 'children': tfMetricsLinks},
		{'title': 'tf.nest', 'collapsabel': false, 'children': tfNestLinks},
		{'title': 'tf.nn', 'collapsabel': false, 'children': tfNNLinks},
		{'title': 'tf.optimizers', 'collapsabel': false, 'children': tfOptimizersLinks},
		{'title': 'tf.quantization', 'collapsabel': false, 'children': tfQuantizationLinks},
		{'title': 'tf.queue', 'collapsabel': false, 'children': tfQueueLinks},
		{'title': 'tf.ragged', 'collapsabel': false, 'children': tfRaggedLinks},
		{'title': 'tf.random', 'collapsabel': false, 'children': tfRandomLinks},
		{'title': 'tf.raw_ops', 'collapsabel': false, 'children': tfRawOpsLinks},
		{'title': 'tf.sets', 'collapsabel': false, 'children': tfSetsLinks},
		{'title': 'tf.signal', 'collapsabel': false, 'children': tfSignalLinks},
		{'title': 'tf.sparse', 'collapsabel': false, 'children': tfSparseLinks},
		{'title': 'tf.strings', 'collapsabel': false, 'children': tfStringsLinks},
		{'title': 'tf.summary', 'collapsabel': false, 'children': tfSummaryLinks},
		{'title': 'tf.sysconfig', 'collapsabel': false, 'children': tfSysconfigLinks},
		{'title': 'tf.test', 'collapsabel': false, 'children': tfTestLinks},
		{'title': 'tf.tpu', 'collapsabel': false, 'children': tfTpuLinks},
		{'title': 'tf.train', 'collapsabel': false, 'children': tfTrainLinks},
		{'title': 'tf.version', 'collapsabel': false, 'children': tfVersionLinks},
		{'title': 'tf.xla', 'collapsabel': false, 'children': tfXlaLinks},
		allSymbolsLinks
	]
};
