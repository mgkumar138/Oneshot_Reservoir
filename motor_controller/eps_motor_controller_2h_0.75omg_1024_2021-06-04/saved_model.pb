©
Ñ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878©¶

motor_controller/h1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_namemotor_controller/h1/kernel

.motor_controller/h1/kernel/Read/ReadVariableOpReadVariableOpmotor_controller/h1/kernel*
_output_shapes
:	*
dtype0

motor_controller/h2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*+
shared_namemotor_controller/h2/kernel

.motor_controller/h2/kernel/Read/ReadVariableOpReadVariableOpmotor_controller/h2/kernel* 
_output_shapes
:
*
dtype0

motor_controller/action/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*/
shared_name motor_controller/action/kernel

2motor_controller/action/kernel/Read/ReadVariableOpReadVariableOpmotor_controller/action/kernel*
_output_shapes
:	(*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

!Adam/motor_controller/h1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!Adam/motor_controller/h1/kernel/m

5Adam/motor_controller/h1/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/motor_controller/h1/kernel/m*
_output_shapes
:	*
dtype0
 
!Adam/motor_controller/h2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!Adam/motor_controller/h2/kernel/m

5Adam/motor_controller/h2/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/motor_controller/h2/kernel/m* 
_output_shapes
:
*
dtype0
§
%Adam/motor_controller/action/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*6
shared_name'%Adam/motor_controller/action/kernel/m
 
9Adam/motor_controller/action/kernel/m/Read/ReadVariableOpReadVariableOp%Adam/motor_controller/action/kernel/m*
_output_shapes
:	(*
dtype0

!Adam/motor_controller/h1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*2
shared_name#!Adam/motor_controller/h1/kernel/v

5Adam/motor_controller/h1/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/motor_controller/h1/kernel/v*
_output_shapes
:	*
dtype0
 
!Adam/motor_controller/h2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*2
shared_name#!Adam/motor_controller/h2/kernel/v

5Adam/motor_controller/h2/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/motor_controller/h2/kernel/v* 
_output_shapes
:
*
dtype0
§
%Adam/motor_controller/action/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	(*6
shared_name'%Adam/motor_controller/action/kernel/v
 
9Adam/motor_controller/action/kernel/v/Read/ReadVariableOpReadVariableOp%Adam/motor_controller/action/kernel/v*
_output_shapes
:	(*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*»
value±B® B§

h1
h2

action
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
^


kernel
	variables
trainable_variables
regularization_losses
	keras_api
^

kernel
	variables
trainable_variables
regularization_losses
	keras_api
^

kernel
	variables
trainable_variables
regularization_losses
	keras_api
v
iter

beta_1

beta_2
	decay
learning_rate
m=m>m?
v@vAvB


0
1
2


0
1
2
 
­
	variables

layers
non_trainable_variables
 layer_regularization_losses
trainable_variables
regularization_losses
!metrics
"layer_metrics
 
TR
VARIABLE_VALUEmotor_controller/h1/kernel$h1/kernel/.ATTRIBUTES/VARIABLE_VALUE


0


0
 
­
	variables

#layers
$non_trainable_variables
%layer_regularization_losses
trainable_variables
regularization_losses
&metrics
'layer_metrics
TR
VARIABLE_VALUEmotor_controller/h2/kernel$h2/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
	variables

(layers
)non_trainable_variables
*layer_regularization_losses
trainable_variables
regularization_losses
+metrics
,layer_metrics
\Z
VARIABLE_VALUEmotor_controller/action/kernel(action/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
	variables

-layers
.non_trainable_variables
/layer_regularization_losses
trainable_variables
regularization_losses
0metrics
1layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 

20
31
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	4total
	5count
6	variables
7	keras_api
D
	8total
	9count
:
_fn_kwargs
;	variables
<	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

40
51

6	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

;	variables
wu
VARIABLE_VALUE!Adam/motor_controller/h1/kernel/m@h1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!Adam/motor_controller/h2/kernel/m@h2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE%Adam/motor_controller/action/kernel/mDaction/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!Adam/motor_controller/h1/kernel/v@h1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE!Adam/motor_controller/h2/kernel/v@h2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE%Adam/motor_controller/action/kernel/vDaction/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1motor_controller/h1/kernelmotor_controller/h2/kernelmotor_controller/action/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_43060658
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
±
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.motor_controller/h1/kernel/Read/ReadVariableOp.motor_controller/h2/kernel/Read/ReadVariableOp2motor_controller/action/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp5Adam/motor_controller/h1/kernel/m/Read/ReadVariableOp5Adam/motor_controller/h2/kernel/m/Read/ReadVariableOp9Adam/motor_controller/action/kernel/m/Read/ReadVariableOp5Adam/motor_controller/h1/kernel/v/Read/ReadVariableOp5Adam/motor_controller/h2/kernel/v/Read/ReadVariableOp9Adam/motor_controller/action/kernel/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_43060780
Ä
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemotor_controller/h1/kernelmotor_controller/h2/kernelmotor_controller/action/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1!Adam/motor_controller/h1/kernel/m!Adam/motor_controller/h2/kernel/m%Adam/motor_controller/action/kernel/m!Adam/motor_controller/h1/kernel/v!Adam/motor_controller/h2/kernel/v%Adam/motor_controller/action/kernel/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_43060844ä
¥/
³
!__inference__traced_save_43060780
file_prefix9
5savev2_motor_controller_h1_kernel_read_readvariableop9
5savev2_motor_controller_h2_kernel_read_readvariableop=
9savev2_motor_controller_action_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop@
<savev2_adam_motor_controller_h1_kernel_m_read_readvariableop@
<savev2_adam_motor_controller_h2_kernel_m_read_readvariableopD
@savev2_adam_motor_controller_action_kernel_m_read_readvariableop@
<savev2_adam_motor_controller_h1_kernel_v_read_readvariableop@
<savev2_adam_motor_controller_h2_kernel_v_read_readvariableopD
@savev2_adam_motor_controller_action_kernel_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ac094245897b482ebd61ad4f2f28bba9/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameè
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ú
valueðBíB$h1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$h2/kernel/.ATTRIBUTES/VARIABLE_VALUEB(action/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB@h1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@h2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDaction/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@h1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@h2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDaction/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names®
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÉ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_motor_controller_h1_kernel_read_readvariableop5savev2_motor_controller_h2_kernel_read_readvariableop9savev2_motor_controller_action_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop<savev2_adam_motor_controller_h1_kernel_m_read_readvariableop<savev2_adam_motor_controller_h2_kernel_m_read_readvariableop@savev2_adam_motor_controller_action_kernel_m_read_readvariableop<savev2_adam_motor_controller_h1_kernel_v_read_readvariableop<savev2_adam_motor_controller_h2_kernel_v_read_readvariableop@savev2_adam_motor_controller_action_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes~
|: :	:
:	(: : : : : : : : : :	:
:	(:	:
:	(: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:&"
 
_output_shapes
:
:%!

_output_shapes
:	(:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:&"
 
_output_shapes
:
:%!

_output_shapes
:	(:%!

_output_shapes
:	:&"
 
_output_shapes
:
:%!

_output_shapes
:	(:

_output_shapes
: 
¹
k
%__inference_h1_layer_call_fn_43060673

inputs
unknown
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_h1_layer_call_and_return_conditional_losses_430605722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý

N__inference_motor_controller_layer_call_and_return_conditional_losses_43060625
input_1
h1_43060581
h2_43060601
action_43060621
identity¢action/StatefulPartitionedCall¢h1/StatefulPartitionedCall¢h2/StatefulPartitionedCallò
h1/StatefulPartitionedCallStatefulPartitionedCallinput_1h1_43060581*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_h1_layer_call_and_return_conditional_losses_430605722
h1/StatefulPartitionedCall
h2/StatefulPartitionedCallStatefulPartitionedCall#h1/StatefulPartitionedCall:output:0h2_43060601*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_h2_layer_call_and_return_conditional_losses_430605922
h2/StatefulPartitionedCall
action/StatefulPartitionedCallStatefulPartitionedCall#h2/StatefulPartitionedCall:output:0action_43060621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_action_layer_call_and_return_conditional_losses_430606122 
action/StatefulPartitionedCallÖ
IdentityIdentity'action/StatefulPartitionedCall:output:0^action/StatefulPartitionedCall^h1/StatefulPartitionedCall^h2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:::2@
action/StatefulPartitionedCallaction/StatefulPartitionedCall28
h1/StatefulPartitionedCallh1/StatefulPartitionedCall28
h2/StatefulPartitionedCallh2/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¾N


$__inference__traced_restore_43060844
file_prefix/
+assignvariableop_motor_controller_h1_kernel1
-assignvariableop_1_motor_controller_h2_kernel5
1assignvariableop_2_motor_controller_action_kernel 
assignvariableop_3_adam_iter"
assignvariableop_4_adam_beta_1"
assignvariableop_5_adam_beta_2!
assignvariableop_6_adam_decay)
%assignvariableop_7_adam_learning_rate
assignvariableop_8_total
assignvariableop_9_count
assignvariableop_10_total_1
assignvariableop_11_count_19
5assignvariableop_12_adam_motor_controller_h1_kernel_m9
5assignvariableop_13_adam_motor_controller_h2_kernel_m=
9assignvariableop_14_adam_motor_controller_action_kernel_m9
5assignvariableop_15_adam_motor_controller_h1_kernel_v9
5assignvariableop_16_adam_motor_controller_h2_kernel_v=
9assignvariableop_17_adam_motor_controller_action_kernel_v
identity_19¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9î
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ú
valueðBíB$h1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$h2/kernel/.ATTRIBUTES/VARIABLE_VALUEB(action/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB@h1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@h2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDaction/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB@h1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB@h2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDaction/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityª
AssignVariableOpAssignVariableOp+assignvariableop_motor_controller_h1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1²
AssignVariableOp_1AssignVariableOp-assignvariableop_1_motor_controller_h2_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¶
AssignVariableOp_2AssignVariableOp1assignvariableop_2_motor_controller_action_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_3¡
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4£
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¢
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ª
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10£
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11£
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12½
AssignVariableOp_12AssignVariableOp5assignvariableop_12_adam_motor_controller_h1_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13½
AssignVariableOp_13AssignVariableOp5assignvariableop_13_adam_motor_controller_h2_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Á
AssignVariableOp_14AssignVariableOp9assignvariableop_14_adam_motor_controller_action_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15½
AssignVariableOp_15AssignVariableOp5assignvariableop_15_adam_motor_controller_h1_kernel_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16½
AssignVariableOp_16AssignVariableOp5assignvariableop_16_adam_motor_controller_h2_kernel_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Á
AssignVariableOp_17AssignVariableOp9assignvariableop_17_adam_motor_controller_action_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpê
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18Ý
Identity_19IdentityIdentity_18:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_19"#
identity_19Identity_19:output:0*]
_input_shapesL
J: ::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ð

@__inference_h2_layer_call_and_return_conditional_losses_43060592

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
ï
#__inference__wrapped_model_43060560
input_16
2motor_controller_h1_matmul_readvariableop_resource6
2motor_controller_h2_matmul_readvariableop_resource:
6motor_controller_action_matmul_readvariableop_resource
identityÊ
)motor_controller/h1/MatMul/ReadVariableOpReadVariableOp2motor_controller_h1_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)motor_controller/h1/MatMul/ReadVariableOp±
motor_controller/h1/MatMulMatMulinput_11motor_controller/h1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
motor_controller/h1/MatMul
motor_controller/h1/ReluRelu$motor_controller/h1/MatMul:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
motor_controller/h1/ReluË
)motor_controller/h2/MatMul/ReadVariableOpReadVariableOp2motor_controller_h2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)motor_controller/h2/MatMul/ReadVariableOpÐ
motor_controller/h2/MatMulMatMul&motor_controller/h1/Relu:activations:01motor_controller/h2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
motor_controller/h2/MatMul
motor_controller/h2/ReluRelu$motor_controller/h2/MatMul:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
motor_controller/h2/ReluÖ
-motor_controller/action/MatMul/ReadVariableOpReadVariableOp6motor_controller_action_matmul_readvariableop_resource*
_output_shapes
:	(*
dtype02/
-motor_controller/action/MatMul/ReadVariableOpÛ
motor_controller/action/MatMulMatMul&motor_controller/h2/Relu:activations:05motor_controller/action/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2 
motor_controller/action/MatMul©
motor_controller/action/SoftmaxSoftmax(motor_controller/action/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2!
motor_controller/action/Softmax}
IdentityIdentity)motor_controller/action/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ::::P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Á
o
)__inference_action_layer_call_fn_43060703

inputs
unknown
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_action_layer_call_and_return_conditional_losses_430606122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

@__inference_h1_layer_call_and_return_conditional_losses_43060666

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø

D__inference_action_layer_call_and_return_conditional_losses_43060612

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
MatMula
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


3__inference_motor_controller_layer_call_fn_43060637
input_1
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_motor_controller_layer_call_and_return_conditional_losses_430606252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ø

D__inference_action_layer_call_and_return_conditional_losses_43060696

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	(*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2
MatMula
SoftmaxSoftmaxMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í

@__inference_h1_layer_call_and_return_conditional_losses_43060572

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß

&__inference_signature_wrapper_43060658
input_1
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_430605602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ(2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
»
k
%__inference_h2_layer_call_fn_43060688

inputs
unknown
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_h2_layer_call_and_return_conditional_losses_430605922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð

@__inference_h2_layer_call_and_return_conditional_losses_43060681

inputs"
matmul_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulY
ReluReluMatMul:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ(tensorflow/serving/predict:ÈZ
Ó
h1
h2

action
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	
signatures
C__call__
*D&call_and_return_all_conditional_losses
E_default_save_signature"ì
_tf_keras_modelÒ{"class_name": "motor_controller", "name": "motor_controller", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "motor_controller"}, "training_config": {"loss": "mean_squared_error", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Û


kernel
	variables
trainable_variables
regularization_losses
	keras_api
F__call__
*G&call_and_return_all_conditional_losses"À
_tf_keras_layer¦{"class_name": "Dense", "name": "h1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "h1", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 5]}}
á

kernel
	variables
trainable_variables
regularization_losses
	keras_api
H__call__
*I&call_and_return_all_conditional_losses"Æ
_tf_keras_layer¬{"class_name": "Dense", "name": "h2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "h2", "trainable": true, "dtype": "float32", "units": 1024, "activation": "relu", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 1024]}}
Ö

kernel
	variables
trainable_variables
regularization_losses
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"»
_tf_keras_layer¡{"class_name": "Dense", "name": "action", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "action", "trainable": true, "dtype": "float32", "units": 40, "activation": "softmax", "use_bias": false, "kernel_initializer": {"class_name": "Zeros", "config": {}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1024}}}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 1024]}}

iter

beta_1

beta_2
	decay
learning_rate
m=m>m?
v@vAvB"
	optimizer
5

0
1
2"
trackable_list_wrapper
5

0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
	variables

layers
non_trainable_variables
 layer_regularization_losses
trainable_variables
regularization_losses
!metrics
"layer_metrics
C__call__
E_default_save_signature
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
,
Lserving_default"
signature_map
-:+	2motor_controller/h1/kernel
'

0"
trackable_list_wrapper
'

0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables

#layers
$non_trainable_variables
%layer_regularization_losses
trainable_variables
regularization_losses
&metrics
'layer_metrics
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
.:,
2motor_controller/h2/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables

(layers
)non_trainable_variables
*layer_regularization_losses
trainable_variables
regularization_losses
+metrics
,layer_metrics
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
1:/	(2motor_controller/action/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
	variables

-layers
.non_trainable_variables
/layer_regularization_losses
trainable_variables
regularization_losses
0metrics
1layer_metrics
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
»
	4total
	5count
6	variables
7	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ
	8total
	9count
:
_fn_kwargs
;	variables
<	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
40
51"
trackable_list_wrapper
-
6	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
80
91"
trackable_list_wrapper
-
;	variables"
_generic_user_object
2:0	2!Adam/motor_controller/h1/kernel/m
3:1
2!Adam/motor_controller/h2/kernel/m
6:4	(2%Adam/motor_controller/action/kernel/m
2:0	2!Adam/motor_controller/h1/kernel/v
3:1
2!Adam/motor_controller/h2/kernel/v
6:4	(2%Adam/motor_controller/action/kernel/v
ü2ù
3__inference_motor_controller_layer_call_fn_43060637Á
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
2
N__inference_motor_controller_layer_call_and_return_conditional_losses_43060625Á
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
á2Þ
#__inference__wrapped_model_43060560¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
Ï2Ì
%__inference_h1_layer_call_fn_43060673¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_h1_layer_call_and_return_conditional_losses_43060666¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_h2_layer_call_fn_43060688¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_h2_layer_call_and_return_conditional_losses_43060681¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_action_layer_call_fn_43060703¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_action_layer_call_and_return_conditional_losses_43060696¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5B3
&__inference_signature_wrapper_43060658input_1
#__inference__wrapped_model_43060560l
0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ(¤
D__inference_action_layer_call_and_return_conditional_losses_43060696\0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 |
)__inference_action_layer_call_fn_43060703O0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ( 
@__inference_h1_layer_call_and_return_conditional_losses_43060666\
/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 x
%__inference_h1_layer_call_fn_43060673O
/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
@__inference_h2_layer_call_and_return_conditional_losses_43060681]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_h2_layer_call_fn_43060688P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ°
N__inference_motor_controller_layer_call_and_return_conditional_losses_43060625^
0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ(
 
3__inference_motor_controller_layer_call_fn_43060637Q
0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ(¡
&__inference_signature_wrapper_43060658w
;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ(