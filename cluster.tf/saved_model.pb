Ș
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68�}
�
ConstConst*
_output_shapes

:*
dtype0*y
valuepBn"`H"�= �˺ ��:޳> �:>�$�>?H����>��[��$W>��ӻˏ?��:jH>;�2�;A� ;�N(;v�=꾻�*�4�.��?�������;
`
Const_1Const*
_output_shapes
:*
dtype0*%
valueB"2�<|@��ٿ�<�?
�
Const_2Const*
_output_shapes

:*
dtype0*I
value@B>"0��I�4�;?ܾ֝�H�>5�X?���L�Or��U%D>V��������
\
Const_3Const*
_output_shapes
:*
dtype0*!
valueB"���IF~?O�
�
Const_4Const*
_output_shapes

:*
dtype0*I
value@B>"0�´�ӿ�+��,������y@���?�:D����[0�c�C�oR��
`
Const_5Const*
_output_shapes
:*
dtype0*%
valueB"�?a�^B�?�䤿09K@
�
Const_6Const*
_output_shapes

:*
dtype0*y
valuepBn"`���@dZ��K�@X����u�@4�Z�RaA�����@OZ�6t�@A+��s!�@]#Z��8�@���E�@V Z���@�"�A���A�h����A
h
Const_7Const*
_output_shapes
:*
dtype0*-
value$B""7K�@TA�N�@��A��@�a�A

NoOpNoOp
�
Const_8Const"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
}
handlers
outputs
initializer_dict
handler_variables
__call__
gen_tensor_dict

signatures* 

* 
* 
* 
* 
* 
* 

	serving_default* 
* 
* 
f
serving_default_inputPlaceholder*
_output_shapes

:*
dtype0*
shape
:
�
PartitionedCallPartitionedCallserving_default_inputConstConst_1Const_2Const_3Const_4Const_5Const_6Const_7*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference_signature_wrapper_96
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *%
f R
__inference__traced_save_182
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_restore_192�j
�%
�
__inference___call___73	
input
transpose_x
mul_1_y
transpose_1_x
mul_3_y
transpose_2_x
mul_5_y
transpose_3_x
mul_7_y
identity^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   b
flatten/ReshapeReshapeinputflatten/Const:output:0*
T0*
_output_shapes

:_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       e
	transpose	Transposetranspose_xtranspose/perm:output:0*
T0*
_output_shapes

:b
MatMulMatMulflatten/Reshape:output:0transpose:y:0*
T0*
_output_shapes

:J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
mulMulmul/x:output:0MatMul:product:0*
T0*
_output_shapes

:L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
mul_1Mulmul_1/x:output:0mul_1_y*
T0*
_output_shapes
:I
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes

:O
onnx_tf_prefix_Tanh_1Tanhadd:z:0*
T0*
_output_shapes

:`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   z
flatten_1/ReshapeReshapeonnx_tf_prefix_Tanh_1:y:0flatten_1/Const:output:0*
T0*
_output_shapes

:a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       k
transpose_1	Transposetranspose_1_xtranspose_1/perm:output:0*
T0*
_output_shapes

:h
MatMul_1MatMulflatten_1/Reshape:output:0transpose_1:y:0*
T0*
_output_shapes

:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_2Mulmul_2/x:output:0MatMul_1:product:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
mul_3Mulmul_3/x:output:0mul_3_y*
T0*
_output_shapes
:M
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:Q
onnx_tf_prefix_Tanh_3Tanh	add_1:z:0*
T0*
_output_shapes

:`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   z
flatten_2/ReshapeReshapeonnx_tf_prefix_Tanh_3:y:0flatten_2/Const:output:0*
T0*
_output_shapes

:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       k
transpose_2	Transposetranspose_2_xtranspose_2/perm:output:0*
T0*
_output_shapes

:h
MatMul_2MatMulflatten_2/Reshape:output:0transpose_2:y:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_4Mulmul_4/x:output:0MatMul_2:product:0*
T0*
_output_shapes

:L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
mul_5Mulmul_5/x:output:0mul_5_y*
T0*
_output_shapes
:M
add_2AddV2	mul_4:z:0	mul_5:z:0*
T0*
_output_shapes

:Q
onnx_tf_prefix_Tanh_5Tanh	add_2:z:0*
T0*
_output_shapes

:`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   z
flatten_3/ReshapeReshapeonnx_tf_prefix_Tanh_5:y:0flatten_3/Const:output:0*
T0*
_output_shapes

:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       k
transpose_3	Transposetranspose_3_xtranspose_3/perm:output:0*
T0*
_output_shapes

:h
MatMul_3MatMulflatten_3/Reshape:output:0transpose_3:y:0*
T0*
_output_shapes

:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_6Mulmul_6/x:output:0MatMul_3:product:0*
T0*
_output_shapes

:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
mul_7Mulmul_7/x:output:0mul_7_y*
T0*
_output_shapes
:M
add_3AddV2	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes

:H
IdentityIdentity	add_3:z:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J::::::::::E A

_output_shapes

:

_user_specified_nameinput:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:
�
E
__inference__traced_restore_192
file_prefix

identity_1��
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
 __inference_signature_wrapper_96	
input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity�
PartitionedCallPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� * 
fR
__inference___call___73W
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J::::::::::E A

_output_shapes

:

_user_specified_nameinput:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:
�%
�
__inference___call___151	
input
transpose_x
mul_1_y
transpose_1_x
mul_3_y
transpose_2_x
mul_5_y
transpose_3_x
mul_7_y
identity^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   b
flatten/ReshapeReshapeinputflatten/Const:output:0*
T0*
_output_shapes

:_
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       e
	transpose	Transposetranspose_xtranspose/perm:output:0*
T0*
_output_shapes

:b
MatMulMatMulflatten/Reshape:output:0transpose:y:0*
T0*
_output_shapes

:J
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?U
mulMulmul/x:output:0MatMul:product:0*
T0*
_output_shapes

:L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
mul_1Mulmul_1/x:output:0mul_1_y*
T0*
_output_shapes
:I
addAddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes

:O
onnx_tf_prefix_Tanh_1Tanhadd:z:0*
T0*
_output_shapes

:`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   z
flatten_1/ReshapeReshapeonnx_tf_prefix_Tanh_1:y:0flatten_1/Const:output:0*
T0*
_output_shapes

:a
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       k
transpose_1	Transposetranspose_1_xtranspose_1/perm:output:0*
T0*
_output_shapes

:h
MatMul_1MatMulflatten_1/Reshape:output:0transpose_1:y:0*
T0*
_output_shapes

:L
mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_2Mulmul_2/x:output:0MatMul_1:product:0*
T0*
_output_shapes

:L
mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
mul_3Mulmul_3/x:output:0mul_3_y*
T0*
_output_shapes
:M
add_1AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes

:Q
onnx_tf_prefix_Tanh_3Tanh	add_1:z:0*
T0*
_output_shapes

:`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   z
flatten_2/ReshapeReshapeonnx_tf_prefix_Tanh_3:y:0flatten_2/Const:output:0*
T0*
_output_shapes

:a
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       k
transpose_2	Transposetranspose_2_xtranspose_2/perm:output:0*
T0*
_output_shapes

:h
MatMul_2MatMulflatten_2/Reshape:output:0transpose_2:y:0*
T0*
_output_shapes

:L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_4Mulmul_4/x:output:0MatMul_2:product:0*
T0*
_output_shapes

:L
mul_5/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
mul_5Mulmul_5/x:output:0mul_5_y*
T0*
_output_shapes
:M
add_2AddV2	mul_4:z:0	mul_5:z:0*
T0*
_output_shapes

:Q
onnx_tf_prefix_Tanh_5Tanh	add_2:z:0*
T0*
_output_shapes

:`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"����   z
flatten_3/ReshapeReshapeonnx_tf_prefix_Tanh_5:y:0flatten_3/Const:output:0*
T0*
_output_shapes

:a
transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       k
transpose_3	Transposetranspose_3_xtranspose_3/perm:output:0*
T0*
_output_shapes

:h
MatMul_3MatMulflatten_3/Reshape:output:0transpose_3:y:0*
T0*
_output_shapes

:L
mul_6/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?[
mul_6Mulmul_6/x:output:0MatMul_3:product:0*
T0*
_output_shapes

:L
mul_7/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?L
mul_7Mulmul_7/x:output:0mul_7_y*
T0*
_output_shapes
:M
add_3AddV2	mul_6:z:0	mul_7:z:0*
T0*
_output_shapes

:H
IdentityIdentity	add_3:z:0*
T0*
_output_shapes

:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J::::::::::E A

_output_shapes

:

_user_specified_nameinput:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
:
�
k
__inference__traced_save_182
file_prefix
savev2_const_8

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const_8"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: "�J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_defaultw
.
input%
serving_default_input:0)
output
PartitionedCall:0tensorflow/serving/predict:�

�
handlers
outputs
initializer_dict
handler_variables
__call__
gen_tensor_dict

signatures"
_generic_user_object
$
"
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
�2�
__inference___call___151�
���
FullArgSpec
args�
jself
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jself
j
input_dict
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
	serving_default"
signature_map
 "
trackable_dict_wrapper
�B�
 __inference_signature_wrapper_96input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7~
__inference___call___151b
.�+
� 
$�!

input�
input"&�#
!
output�
output�
 __inference_signature_wrapper_96b
.�+
� 
$�!

input�
input"&�#
!
output�
output