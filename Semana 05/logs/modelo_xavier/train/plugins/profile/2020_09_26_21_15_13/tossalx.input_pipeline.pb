	><K�P�?><K�P�?!><K�P�?	':XV�.@':XV�.@!':XV�.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$><K�P�?4�IbI��?A�a�� ��?Y�������?*	T㥛�P@2F
Iterator::Model:vP��?!��fS�M@)��*��]�?1���5��H@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate�f�|��?!������5@)�\��?1���!</2@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeatT5A�} �?!O���h+@)j���<��?1�d�m!)@:Preprocessing2S
Iterator::Model::ParallelMap��x�[y?!��v8N#@)��x�[y?1��v8N#@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip_{fI���?!Y9���@D@)��ŉ�vd?1]�v�(@:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��C���b?!ێi��@)��C���b?1ێi��@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�G���?!g�V1?�6@)�*5{�H?1�Mܶ�U�?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor����G?!�E�h<�?)����G?1�E�h<�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 15.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2B20.2 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	4�IbI��?4�IbI��?!4�IbI��?      ��!       "      ��!       *      ��!       2	�a�� ��?�a�� ��?!�a�� ��?:      ��!       B      ��!       J	�������?�������?!�������?R      ��!       Z	�������?�������?!�������?JCPU_ONLY